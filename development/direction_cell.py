# ============ ONE CELL: direction test for paper 2. Does OmniSQL-7B work without NIFT-SQL? BIRD dev, A100. Fresh runtime -> paste -> run. ============
# Four schemas given to the same generator (OmniSQL-7B, greedy), execution accuracy on BIRD dev:
#   1 gold database, full schema        (the generator's ceiling; its paper reports 63.9)
#   2 gold tables only                  (perfect table retrieval)
#   3 routed database, full schema      (plain hybrid search picks the database; no NIFT-SQL)
#   4 routed database, top-15 tables    (plain hybrid search picks the tables; no NIFT-SQL)
# Direction only: the catalog is BIRD dev's 11 databases (75 tables), not RASL's 597-table pooled catalog.
# About 30 to 45 minutes; outputs and the report go to OUT_DIR, and finished conditions are reused after a restart.
OUT_DIR = "/content/drive/MyDrive/nift_rasl/direction"
N_QUESTIONS = 0          # 0 = all 1,534; e.g. 300 for a quicker look (same random subset every time)
MODEL = "seeklhy/OmniSQL-7B"
TOP_K = 15
# =====================================================================================================================
import os, subprocess, sys, time
T0 = time.time()
def say(msg): print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)
say("step 1/6: installing vLLM and helpers (3 to 5 minutes, quiet)")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "rank_bm25", "sentence-transformers", "func_timeout"], check=True)
say("installed")
from google.colab import drive
drive.mount("/content/drive")
import json, re, sqlite3, zipfile, random, time, glob
from pathlib import Path
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut
OUT = Path(OUT_DIR); OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("/content/bird_dev")

# ---------------------------------------------------------------- 1  BIRD dev: questions and sqlite databases
if not (DATA / "dev.json").exists():
    DATA.mkdir(exist_ok=True)
    say("step 2/6: downloading BIRD dev (about 1.4 GB)")
    subprocess.run(["wget", "-q", "--show-progress", "--progress=bar:force:noscroll", "-O", "/content/dev.zip",
                    "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"], check=True)
    say("unzipping")
    with zipfile.ZipFile("/content/dev.zip") as z: z.extractall("/content/bird_raw")
    root = Path(glob.glob("/content/bird_raw/**/dev.json", recursive=True)[0]).parent
    for f in ("dev.json", "dev_tables.json"): (DATA / f).write_bytes((root / f).read_bytes())
    inner = glob.glob(str(root / "dev_databases*.zip"))
    if inner:
        with zipfile.ZipFile(inner[0]) as z: z.extractall(DATA)
    else:
        subprocess.run(["cp", "-r", str(root / "dev_databases"), str(DATA)], check=True)
say("BIRD dev ready")
DBDIR = Path(glob.glob(str(DATA / "**/dev_databases"), recursive=True)[0])
Q = json.loads((DATA / "dev.json").read_text())
if N_QUESTIONS:
    Q = random.Random(0).sample(Q, N_QUESTIONS)
dbpath = lambda db: str(DBDIR / db / f"{db}.sqlite")

# ---------------------------------------------------------------- 2  schemas: DDL per table, straight from sqlite
DDL, COLS = {}, {}
for db in sorted({q["db_id"] for q in json.loads((DATA / "dev.json").read_text())}):
    con = sqlite3.connect(dbpath(db))
    for name, sql in con.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        DDL[(db, name)] = sql.strip()
        COLS[(db, name)] = [r[1] for r in con.execute(f'PRAGMA table_info("{name}")')]
    con.close()
TABLES = sorted(DDL)
print(f"catalog: {len({d for d, _ in TABLES})} databases, {len(TABLES)} tables | questions: {len(Q)}")


def gold_tables(q):
    """Tables named in the gold SQL (case-insensitive match against the gold database's tables)."""
    sql = q["SQL"].lower()
    return [t for d, t in TABLES if d == q["db_id"] and re.search(r'(?<![\w])[`"\[]?' + re.escape(t.lower()) + r'[`"\]]?(?![\w])', sql)]


# ---------------------------------------------------------------- 3  plain hybrid search (no training): 0.6 dense BGE + 0.4 BM25
say("step 3/6: plain hybrid search (BM25 + BGE) over the catalog")
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
words = lambda s: re.findall(r"[a-z0-9]+", s.lower().replace("_", " "))
cards = [f"{t}: " + ", ".join(COLS[(d, t)]) for d, t in TABLES]
qtext = [q["question"] + " " + q.get("evidence", "") for q in Q]
bm = BM25Okapi([words(c) for c in cards])
BM = np.stack([bm.get_scores(words(x)) for x in qtext])
emb = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
DE = emb.encode(qtext, normalize_embeddings=True, batch_size=64) @ emb.encode(cards, normalize_embeddings=True).T
del emb
import torch, gc; gc.collect(); torch.cuda.empty_cache()
mm = lambda X: (X - X.min(1, keepdims=True)) / (X.max(1, keepdims=True) - X.min(1, keepdims=True) + 1e-9)
HY = 0.6 * mm(DE) + 0.4 * mm(BM)

route, top_tables, recall, complete, routed_ok = [], [], [], [], []
for i, q in enumerate(Q):
    order = np.argsort(-HY[i])
    db = TABLES[order[0]][0]                                     # route to the database of the best-ranked table
    route.append(db)
    tops = [TABLES[j][1] for j in order if TABLES[j][0] == db][:TOP_K]
    top_tables.append(tops)
    g = set(gold_tables(q)); routed_ok.append(db == q["db_id"])
    hit = g & set(tops) if db == q["db_id"] else set()
    recall.append(len(hit) / max(1, len(g))); complete.append(float(g <= hit and bool(g)))
print(f"plain hybrid search: routing accuracy {100*np.mean(routed_ok):.1f}% | table recall@{TOP_K} {100*np.mean(recall):.1f}% "
      f"| complete@{TOP_K} {100*np.mean(complete):.1f}%")

# ---------------------------------------------------------------- 4  OmniSQL prompts (format from the model card)
TEMPLATE = """Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{schema}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query."""
schema_of = lambda db, tabs: "\n\n".join(DDL[(db, t)] for t in tabs if (db, t) in DDL)
all_tabs = lambda db: [t for d, t in TABLES if d == db]
CONDS = {
    "1 gold database, full schema": [(q["db_id"], all_tabs(q["db_id"])) for q in Q],
    "2 gold tables only": [(q["db_id"], gold_tables(q) or all_tabs(q["db_id"])) for q in Q],
    "3 routed database, full schema": [(route[i], all_tabs(route[i])) for i in range(len(Q))],
    f"4 routed database, top-{TOP_K} tables": [(route[i], top_tables[i]) for i in range(len(Q))],
}

# ---------------------------------------------------------------- 5  generate (vLLM, greedy) and execute
say("step 4/6: loading OmniSQL-7B into vLLM (2 to 4 minutes)")
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(MODEL, max_model_len=12288, gpu_memory_utilization=0.88, seed=0)
sp = SamplingParams(temperature=0.0, max_tokens=2048)


def extract(text):
    blocks = re.findall(r"```sql\s*(.*?)```", text, re.S)
    return (blocks[-1] if blocks else text).strip()


def run(db, sql):
    con = sqlite3.connect(dbpath(db)); con.text_factory = lambda b: b.decode(errors="ignore")
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def correct(db_pred, sql_pred, q):
    try:
        gold = func_timeout(30, run, args=(q["db_id"], q["SQL"]))
        pred = func_timeout(30, run, args=(db_pred, sql_pred))
        return float(set(pred) == set(gold))                    # the official BIRD comparison
    except (FunctionTimedOut, Exception):
        return 0.0


say("step 5/6: generating SQL for 4 conditions (progress bars below; about 5 to 10 minutes each)")
report = {}
for name, plan in CONDS.items():
    f = OUT / (re.sub(r"\W+", "_", name) + ".jsonl")
    if f.exists() and sum(1 for _ in open(f)) == len(Q):
        rows = [json.loads(l) for l in open(f)]
    else:
        t0 = time.time(); say(f"condition {name}: generating")
        prompts = [tok.apply_chat_template([{"role": "user", "content": TEMPLATE.format(
                   schema=schema_of(db, tabs), question=(q["question"] + "\n" + q.get("evidence", "")).strip())}],
                   tokenize=False, add_generation_prompt=True) for q, (db, tabs) in zip(Q, plan)]
        outs = llm.generate(prompts, sp, use_tqdm=True)
        say(f"condition {name}: executing {len(outs)} queries against sqlite")
        rows = []
        for q, (db, tabs), o in zip(Q, plan, outs):
            sql = extract(o.outputs[0].text)
            rows.append(dict(question_id=q.get("question_id"), db=db, gold_db=q["db_id"], difficulty=q.get("difficulty", ""),
                             n_tables=len(tabs), sql=sql, ok=correct(db, sql, q)))
        with open(f, "w") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{name}: {len(rows)} questions in {(time.time() - t0) / 60:.1f} min", flush=True)
    ok = np.array([r["ok"] for r in rows])
    by = {d: 100 * np.mean([r["ok"] for r in rows if r["difficulty"] == d]) for d in ("simple", "moderate", "challenging")
          if any(r["difficulty"] == d for r in rows)}
    report[name] = (100 * ok.mean(), np.mean([r["n_tables"] for r in rows]), by)
    say(f"condition {name}: EX {100 * ok.mean():.1f}%")

say("step 6/6: report")
L = [f"Direction test: OmniSQL-7B (greedy) on BIRD dev, {len(Q)} questions | catalog {len(TABLES)} tables in "
     f"{len({d for d, _ in TABLES})} databases | plain hybrid search, no NIFT-SQL",
     f"routing accuracy {100*np.mean(routed_ok):.1f}% | table recall@{TOP_K} {100*np.mean(recall):.1f}% | complete@{TOP_K} {100*np.mean(complete):.1f}%", "",
     f"{'condition':40s} {'EX %':>6s} {'tables shown':>13s}   by difficulty"]
for name, (ex, nt, by) in report.items():
    L.append(f"{name:40s} {ex:6.1f} {nt:13.1f}   " + " | ".join(f"{d} {v:.1f}" for d, v in by.items()))
L += ["", "For reference (published): OmniSQL-7B 63.9 EX with the gold database; RASL (Claude 3.5 Sonnet-v2) 53.5 EX, 54.0 with gold tables."]
(OUT / "DIRECTION.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
