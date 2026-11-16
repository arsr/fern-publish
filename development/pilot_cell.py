# ============ ONE CELL: paper-2 pilot. Does the full pipeline work before we scale? A100 80GB. Fresh runtime -> paste -> run. ============
# Pipeline (only the decider is trained):
#   hybrid search (BM25 + Qwen3-Embedding, RRF) top 100  ─┐
#                                                         ├─ union → ensemble reranker (RF + LightGBM + LR, stacked) → top 30
#   OpenDecider-nano (fine-tuned on BIRD train) top 100  ─┘
#   → Qwen3-8B (thinking) picks up to 15 tables → Gemma-4-31B-it (few-shot) writes the SQL → execution accuracy on BIRD dev
# Catalog: every BIRD train + dev database pooled (RASL's BIRD setting). Dev databases are never seen in training.
# Small subset by default; every stage is cached in OUT_DIR, so a restart resumes where it stopped.
# About 20 minutes retrieval-only; add 15 to 20 minutes with RUN_GENERATION = True.
OUT_DIR = "/content/drive/MyDrive/nift_rasl/pilot"
N_TRAIN_Q = 1000          # BIRD train questions used to fine-tune the decider (0 = all, about 8,500)
N_VAL_Q = 150             # questions on held-out train databases, used to fit the reranker
N_DEV_Q = 100             # BIRD dev questions evaluated (0 = all 1,534)
RUN_GENERATION = False    # False = retrieval only (stops after Qwen3-8B picks tables; skips the 62 GB Gemma download), about 20 min
K_FIRST, K_RERANK, K_FINAL = 100, 30, 15
LOSS = "bce"              # "mix" = BCE + listwise (multi-positive InfoNCE per question); "bce" = BCE only
LAMBDA = 1.0              # weight of the listwise term in "mix"
EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"
DECIDER = "manjunathshiva/opendecider-nano"
PICKER = "Qwen/Qwen3-8B"
GENERATOR = "google/gemma-4-31B-it"
# =====================================================================================================================
import os, subprocess, sys, time
T0 = time.time()
def say(msg): print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)
say("step 1/9: installing (3 to 5 minutes, quiet)")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "rank_bm25", "sentence-transformers", "func_timeout",
                "lightgbm", "remotezip"], check=True)
from google.colab import drive
drive.mount("/content/drive")
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
import json, re, sqlite3, zipfile, random, glob
from pathlib import Path
import numpy as np
OUT = Path(OUT_DIR); OUT.mkdir(parents=True, exist_ok=True)
TAG = f"t{N_TRAIN_Q}_v{N_VAL_Q}_d{N_DEV_Q}_{LOSS}"
TIMES = {}

# ---------------------------------------------------------------- 2  BIRD data: dev (with sqlite) and train (json only)
say("step 2/9: BIRD dev and train")
DEV = Path("/content/bird_dev"); TRN = Path("/content/bird_train")
if not (DEV / "dev.json").exists():
    DEV.mkdir(exist_ok=True)
    subprocess.run(["wget", "-q", "--show-progress", "--progress=bar:force:noscroll", "-O", "/content/dev.zip",
                    "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"], check=True)
    with zipfile.ZipFile("/content/dev.zip") as z: z.extractall("/content/bird_raw")
    root = Path(glob.glob("/content/bird_raw/**/dev.json", recursive=True)[0]).parent
    for f in ("dev.json", "dev_tables.json"): (DEV / f).write_bytes((root / f).read_bytes())
    inner = glob.glob(str(root / "dev_databases*.zip"))
    if inner:
        with zipfile.ZipFile(inner[0]) as z: z.extractall(DEV)
    else:
        subprocess.run(["cp", "-r", str(root / "dev_databases"), str(DEV)], check=True)
if not (TRN / "train.json").exists():
    TRN.mkdir(exist_ok=True)
    URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
    try:                                                   # fetch only the two json files from the remote zip
        from remotezip import RemoteZip
        with RemoteZip(URL) as z:
            for want in ("train.json", "train_tables.json"):
                name = [n for n in z.namelist() if n.endswith("/" + want) or n == want][0]
                (TRN / want).write_bytes(z.read(name))
    except Exception as e:
        say(f"remote zip read failed ({type(e).__name__}); downloading the full train.zip instead (several GB)")
        subprocess.run(["wget", "-q", "--show-progress", "--progress=bar:force:noscroll", "-O", "/content/train.zip", URL], check=True)
        with zipfile.ZipFile("/content/train.zip") as z:
            for want in ("train.json", "train_tables.json"):
                name = [n for n in z.namelist() if n.endswith("/" + want) or n == want][0]
                (TRN / want).write_bytes(z.read(name))
        os.remove("/content/train.zip")
DBDIR = Path(glob.glob(str(DEV / "**/dev_databases"), recursive=True)[0])
dbpath = lambda db: str(DBDIR / db / f"{db}.sqlite")
DEV_Q_ALL = json.loads((DEV / "dev.json").read_text())
TRN_Q_ALL = json.loads((TRN / "train.json").read_text())
SCHEMAS = json.loads((TRN / "train_tables.json").read_text()) + json.loads((DEV / "dev_tables.json").read_text())
DEV_DBS = {s["db_id"] for s in json.loads((DEV / "dev_tables.json").read_text())}

# ---------------------------------------------------------------- 3  pooled catalog: one card and one schema block per table
TABLES, CARD, BLOCK = [], [], []
for s in SCHEMAS:
    db, tn = s["db_id"], s["table_names_original"]
    cols = [(t, c, nat, ty) for (t, c), (_, nat), ty in zip(s["column_names_original"], s["column_names"], s["column_types"]) if t >= 0]
    cid = {i: (t, c) for i, (t, c) in enumerate(s["column_names_original"])}
    pk = set()
    for p in s["primary_keys"]:
        pk |= set(p) if isinstance(p, list) else {p}
    for ti, t in enumerate(tn):
        mine = [(i, c, nat, ty) for i, (tt, c, nat, ty) in enumerate(cols, start=1) if tt == ti]
        desc = [c if nat.lower().replace(" ", "") == c.lower().replace("_", "") else f"{c} ({nat})" for _, c, nat, _ in mine]
        fks = [f"{cid[a][1]} -> {tn[cid[b][0]]}.{cid[b][1]}" for a, b in s["foreign_keys"] if cid[a][0] == ti]
        TABLES.append((db, t))
        CARD.append(f"{t} (database {db.replace('_', ' ')}): " + ", ".join(desc))
        BLOCK.append(f"## {db}.{t}\ncolumns: " + ", ".join(f"{c} {ty}{' PK' if i in pk else ''}" for i, c, _, ty in mine)
                     + (f"\nforeign keys: " + "; ".join(fks) if fks else ""))
TID = {t: i for i, t in enumerate(TABLES)}
TBL_OF = {}
for i, (d, t) in enumerate(TABLES): TBL_OF.setdefault(d, []).append(i)
print(f"catalog: {len(TBL_OF)} databases, {len(TABLES)} tables ({sum(len(TBL_OF[d]) for d in DEV_DBS)} in dev databases)")


def gold_ids(q):
    sql = q["SQL"].lower()
    return [i for i in TBL_OF[q["db_id"]]
            if re.search(r'(?<![\w])[`"\[]?' + re.escape(TABLES[i][1].lower()) + r'[`"\]]?(?![\w])', sql)]


qtext = lambda q: (q["question"] + " " + q.get("evidence", "")).strip()
# splits: decider-fit questions, reranker-fit questions on held-out train databases, dev questions
rng = random.Random(0)
trn_dbs = sorted({q["db_id"] for q in TRN_Q_ALL}); rng.shuffle(trn_dbs)
VAL_DBS = set(trn_dbs[:max(5, len(trn_dbs) // 10)])
fitpool = [q for q in TRN_Q_ALL if q["db_id"] not in VAL_DBS and gold_ids(q)]
valpool = [q for q in TRN_Q_ALL if q["db_id"] in VAL_DBS and gold_ids(q)]
FIT = random.Random(1).sample(fitpool, N_TRAIN_Q) if N_TRAIN_Q and N_TRAIN_Q < len(fitpool) else fitpool
VAL = random.Random(2).sample(valpool, min(N_VAL_Q, len(valpool))) if N_VAL_Q else valpool
DEVQ = random.Random(3).sample(DEV_Q_ALL, N_DEV_Q) if N_DEV_Q else DEV_Q_ALL
print(f"decider fit: {len(FIT)} questions | reranker fit: {len(VAL)} questions on {len(VAL_DBS)} held-out train databases | dev: {len(DEVQ)}")

# ---------------------------------------------------------------- 4  hybrid search: BM25 + Qwen3-Embedding, reciprocal rank fusion
say("step 3/9: hybrid search (BM25 + Qwen3-Embedding, RRF)")
t0 = time.time()
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch, gc
words = lambda s: re.findall(r"[a-z0-9]+", s.lower().replace("_", " "))
bm = BM25Okapi([words(c) for c in CARD])
emb = SentenceTransformer(EMBEDDER, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
CE = emb.encode(CARD, normalize_embeddings=True, batch_size=32)
POOL = TRN_Q_ALL                                                          # few-shot example pool (train only)
PE = emb.encode([qtext(q) for q in POOL], prompt_name="query", normalize_embeddings=True, batch_size=64)


def hybrid(qs):
    qe = emb.encode([qtext(q) for q in qs], prompt_name="query", normalize_embeddings=True, batch_size=64)
    DE = qe @ CE.T
    BM = np.stack([bm.get_scores(words(qtext(q))) for q in qs])
    rank = lambda X: np.argsort(np.argsort(-X, 1), 1)
    return 1 / (60 + rank(DE)) + 1 / (60 + rank(BM)), DE, BM, qe


H = {name: hybrid(qs) for name, qs in (("fit", FIT), ("val", VAL), ("dev", DEVQ))}
del emb; gc.collect(); torch.cuda.empty_cache()
TIMES["hybrid search"] = time.time() - t0

# ---------------------------------------------------------------- 5  decider: OpenDecider-nano, fine-tuned as a yes/no table classifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
DEC_DIR = OUT / f"decider_{TAG}"


def load_decider(path):
    try:
        m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, trust_remote_code=True, ignore_mismatched_sizes=True)
        return m, AutoTokenizer.from_pretrained(path, trust_remote_code=True), path
    except Exception as e:
        say(f"!! could not load {path} as a classifier ({type(e).__name__}: {str(e)[:200]}). FALLING BACK to answerdotai/ModernBERT-large")
        p = "answerdotai/ModernBERT-large"
        return AutoModelForSequenceClassification.from_pretrained(p, num_labels=1), AutoTokenizer.from_pretrained(p), p


if (DEC_DIR / "config.json").exists():
    say("step 4/9: decider already fine-tuned, loading it")
    model, tok, used = load_decider(str(DEC_DIR)); used = (DEC_DIR / "base.txt").read_text()
else:
    say(f"step 4/9: fine-tuning the decider on {len(FIT)} questions")
    t0 = time.time()
    model, tok, used = load_decider(DECIDER)
    # one group per question: its gold tables (label 1), 8 hard negatives from search, 4 random negatives (label 0)
    S = H["fit"][0]; groups = []
    r = random.Random(4)
    for i, q in enumerate(FIT):
        g = set(gold_ids(q))
        hard = [j for j in np.argsort(-S[i])[:40] if j not in g][:8]          # hard negatives: what search ranks high
        easy = r.sample([j for j in range(len(TABLES)) if j not in g], 4)
        groups.append([(qtext(q), CARD[j], 1.0) for j in g] + [(qtext(q), CARD[j], 0.0) for j in hard + easy])
    r.shuffle(groups)
    model.cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    QPB = 2; steps = (len(groups) + QPB - 1) // QPB                         # 2 questions per batch, about 28 pairs
    sch = get_linear_schedule_with_warmup(opt, steps // 10, steps)
    bce = torch.nn.BCEWithLogitsLoss(); run = 0.0


    def listwise(logit, y):
        """Multi-positive InfoNCE for one question: each gold table must outscore all of that question's negatives."""
        pos, neg = logit[y > 0.5], logit[y < 0.5]
        if not len(pos) or not len(neg): return logit.sum() * 0
        return (torch.logsumexp(torch.cat([pos[:, None], neg[None, :].expand(len(pos), -1)], 1), 1) - pos).mean()


    for s in range(steps):
        gb = groups[s * QPB:(s + 1) * QPB]; b = [p for grp in gb for p in grp]
        x = tok([p[0] for p in b], [p[1] for p in b], truncation=True, max_length=384, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(**x).logits[:, 0].float()
        y = torch.tensor([p[2] for p in b], device="cuda")
        loss = bce(logit, y)
        if LOSS == "mix":
            cuts = np.cumsum([0] + [len(grp) for grp in gb])
            loss = loss + LAMBDA * torch.stack([listwise(logit[a:c], y[a:c]) for a, c in zip(cuts[:-1], cuts[1:])]).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad()
        run = 0.98 * run + 0.02 * loss.item() if s else loss.item()
        if s % 200 == 0 or s == steps - 1:
            say(f"  decider step {s + 1}/{steps}  loss ({LOSS}) {run:.4f}")
    model.save_pretrained(DEC_DIR); tok.save_pretrained(DEC_DIR); (DEC_DIR / "base.txt").write_text(used)
    TIMES["decider fine-tune"] = time.time() - t0
print(f"decider base: {used}")


@torch.no_grad()
def decide(qs, name):
    f = OUT / f"dec_{name}_{TAG}.npy"
    if f.exists(): return np.load(f)
    model.cuda().eval(); M = np.zeros((len(qs), len(TABLES)), dtype=np.float32)
    for i, q in enumerate(qs):
        for a in range(0, len(TABLES), 320):
            x = tok([qtext(q)] * len(CARD[a:a + 320]), CARD[a:a + 320], truncation=True, max_length=384, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                M[i, a:a + 320] = model(**x).logits[:, 0].float().cpu().numpy()
        if i % 100 == 0: say(f"  decider scoring {name}: {i}/{len(qs)}")
    np.save(f, M); return M


say("step 5/9: decider scores every table in the catalog for each question")
t0 = time.time()
DEC = {"val": decide(VAL, "val"), "dev": decide(DEVQ, "dev")}
TIMES["decider scoring (val + dev)"] = time.time() - t0
del model; gc.collect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------- 6  union of the two top-100 lists, features, stacked ensemble
say("step 6/9: union and ensemble reranker")
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
rank = lambda X: np.argsort(np.argsort(-X, 1), 1)


def build(name, qs):
    S, DE, BM, _ = H[name]; D = DEC[name]
    RH, RD = rank(S), rank(D)
    BMn = (BM - BM.min(1, keepdims=True)) / (BM.max(1, keepdims=True) - BM.min(1, keepdims=True) + 1e-9)
    X, y, grp, cand = [], [], [], []
    for i, q in enumerate(qs):
        top_h = set(np.argsort(-S[i])[:K_FIRST]); top_d = set(np.argsort(-D[i])[:K_FIRST]); U = sorted(top_h | top_d)
        g = set(gold_ids(q)); dbs = [TABLES[j][0] for j in U]
        db_max_d, db_max_h, db_n = {}, {}, {}
        for j, d in zip(U, dbs):
            db_max_d[d] = max(db_max_d.get(d, -1e9), D[i, j]); db_max_h[d] = max(db_max_h.get(d, 0), S[i, j]); db_n[d] = db_n.get(d, 0) + 1
        order_db = {d: r for r, d in enumerate(sorted(db_max_d, key=lambda d: -db_max_d[d]))}
        for j, d in zip(U, dbs):
            X.append([S[i, j], RH[i, j], DE[i, j], BMn[i, j], D[i, j], RD[i, j], j in top_h, j in top_d, (j in top_h) and (j in top_d),
                      db_n[d], db_max_d[d], db_max_h[d], order_db[d], D[i, j] - db_max_d[d], len(TBL_OF[d])])
            y.append(j in g); grp.append(i); cand.append(j)
    return np.array(X, dtype=np.float32), np.array(y), np.array(grp), np.array(cand)


Xv, yv, _, _ = build("val", VAL)
Xd, yd, gd, cd = build("dev", DEVQ)
ens = StackingClassifier(estimators=[
    ("rf", RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=0)),
    ("lgb", LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1)),
    ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)))],
    final_estimator=LogisticRegression(max_iter=3000), cv=5, n_jobs=1)
ens.fit(Xv, yv)
pd_ = ens.predict_proba(Xd)[:, 1]
RER = []                                                                  # reranked candidate list per dev question
for i in range(len(DEVQ)):
    m = gd == i
    RER.append(list(cd[m][np.argsort(-pd_[m])]))


def rec(sets):
    r, c = [], []
    for q, s in zip(DEVQ, sets):
        g = set(gold_ids(q)); h = g & set(s)
        r.append(len(h) / len(g)); c.append(float(h == g))
    return 100 * np.mean(r), 100 * np.mean(c)


S_dev, D_dev = H["dev"][0], DEC["dev"]
top = lambda M, i, k: list(np.argsort(-M[i])[:k])
RRF2 = 1 / (60 + rank(S_dev)) + 1 / (60 + rank(D_dev))
R = {}
R[f"hybrid search, top {K_FIRST}"] = rec([top(S_dev, i, K_FIRST) for i in range(len(DEVQ))])
R[f"decider, top {K_FIRST}"] = rec([top(D_dev, i, K_FIRST) for i in range(len(DEVQ))])
R[f"union of both top {K_FIRST}"] = rec([set(top(S_dev, i, K_FIRST)) | set(top(D_dev, i, K_FIRST)) for i in range(len(DEVQ))])
only_h = only_d = 0
for i, q in enumerate(DEVQ):
    g = set(gold_ids(q)); a = g & set(top(S_dev, i, K_FIRST)); b = g & set(top(D_dev, i, K_FIRST))
    only_h += len(a - b); only_d += len(b - a)
for k in (K_RERANK, K_FINAL):
    R[f"hybrid search, top {k}"] = rec([top(S_dev, i, k) for i in range(len(DEVQ))])
    R[f"decider, top {k}"] = rec([top(D_dev, i, k) for i in range(len(DEVQ))])
    R[f"RRF of hybrid + decider (no learning), top {k}"] = rec([top(RRF2, i, k) for i in range(len(DEVQ))])
    R[f"ensemble reranker, top {k}"] = rec([RER[i][:k] for i in range(len(DEVQ))])

# ---------------------------------------------------------------- 7  vLLM runner (separate process, so GPU memory is freed after each model)
RUNNER = r'''
import json, sys, os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
a = json.loads(open(sys.argv[1]).read())
tok = AutoTokenizer.from_pretrained(a["model"])
rows = [json.loads(l) for l in open(a["inp"])]
prompts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, **a["chat_kwargs"]) for r in rows]
limit = a["max_model_len"] - a["max_tokens"]
ok = [i for i, p in enumerate(prompts) if len(tok(p)["input_ids"]) <= limit]
print(f"[runner] {len(prompts)} prompts, {len(prompts) - len(ok)} too long (skipped)", flush=True)
print(f"[runner] loading {a['model']} with {a['llm_kwargs'] or 'default settings'}", flush=True)
llm = LLM(model=a["model"], max_model_len=a["max_model_len"], gpu_memory_utilization=a["gpu"], seed=0, **a["llm_kwargs"])
sp = SamplingParams(temperature=a["temp"], top_p=a["top_p"], top_k=a["top_k"], max_tokens=a["max_tokens"], seed=0)
text, fin = [""] * len(prompts), ["too_long"] * len(prompts)
for c in range(0, len(ok), 128):
    idx = ok[c:c + 128]
    for i, o in zip(idx, llm.generate([prompts[i] for i in idx], sp, use_tqdm=False)):
        text[i], fin[i] = o.outputs[0].text, o.outputs[0].finish_reason
    print(f"[runner] generated {min(c + 128, len(ok))}/{len(ok)}", flush=True)
with open(a["out"], "w") as f:
    for t, r in zip(text, fin): f.write(json.dumps({"text": t, "finish": r}) + "\n")
'''
Path("/content/vllm_run.py").write_text(RUNNER)


CAUSE = re.compile(r"(OutOfMemory|out of memory|CUDA error|ValueError|RuntimeError|AssertionError|KeyError|not supported|"
                   r"Free memory|KV cache|No available memory|max_model_len|Unrecognized|Error:)", re.I)


def llm_run(name, model, msgs, chat_kwargs, max_model_len, max_tokens, temp, top_p, top_k, gpu, attempts=({},)):
    """Run one model in its own process. `attempts` are vLLM settings tried in order until one starts (e.g. FP8 first)."""
    out = OUT / f"{name}_{TAG}.jsonl"
    if out.exists() and sum(1 for _ in open(out)) == len(msgs):
        say(f"  {name}: cached"); return [json.loads(l) for l in open(out)]
    inp = Path(f"/content/{name}_in.jsonl"); inp.write_text("".join(json.dumps({"messages": m}) + "\n" for m in msgs))
    for k, extra in enumerate(attempts, 1):
        cfg = Path(f"/content/{name}_cfg.json")
        cfg.write_text(json.dumps(dict(model=model, inp=str(inp), out=str(out), chat_kwargs=chat_kwargs, max_model_len=max_model_len,
                                       max_tokens=max_tokens, temp=temp, top_p=top_p, top_k=top_k, gpu=gpu, llm_kwargs=extra)))
        log = OUT / f"{name}_log_attempt{k}.txt"
        p = subprocess.Popen([sys.executable, "/content/vllm_run.py", str(cfg)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env={**os.environ, "VLLM_LOGGING_LEVEL": "WARNING", "TOKENIZERS_PARALLELISM": "false"})
        lines = []
        for line in p.stdout:
            lines.append(line.rstrip())
            if line.startswith("[runner]"): say("  " + line.rstrip())
        log.write_text("\n".join(lines))
        if p.wait() == 0:
            return [json.loads(l) for l in open(out)]
        causes = [l for l in lines if CAUSE.search(l) and not l.lstrip().startswith(("File ", "raise", "^", "~"))]
        say(f"  {name}: attempt {k}/{len(attempts)} failed ({extra or 'default settings'}). Likely cause:")
        print("\n".join("      " + l[:300] for l in (causes[-8:] or lines[-15:])), flush=True)
        print(f"      full log: {log}", flush=True)
    raise SystemExit(f"{name} failed on every attempt; the logs are in {OUT}")


# ---------------------------------------------------------------- 8  Qwen3-8B (thinking) picks up to 15 tables from the reranked 30
say(f"step 7/9: {PICKER} picks the tables (loading takes 2 to 3 minutes)")
t0 = time.time()
PICK_PROMPT = """You are selecting database tables for a text-to-SQL system.

Question: {q}
Evidence: {e}

Candidate tables (from several databases; each is named database.table):

{blocks}

Choose the tables needed to write the SQL query that answers the question, including tables that are needed only to join others.
All chosen tables must come from ONE database. List the needed tables first, then any other tables from that database that might
be useful, most useful first, at most {k} in total.
Answer with a JSON list of names only, for example ["db.table_a", "db.table_b"]."""
msgs = [[{"role": "user", "content": PICK_PROMPT.format(q=q["question"], e=q.get("evidence", "") or "none", k=K_FINAL,
                                                       blocks="\n\n".join(BLOCK[j] for j in RER[i][:K_RERANK]))}] for i, q in enumerate(DEVQ)]
picks_raw = llm_run("picker", PICKER, msgs, {"enable_thinking": True}, 16384, 6144, 0.6, 0.95, 20, 0.85)
TIMES["picker"] = time.time() - t0


def parse_pick(text, cands):
    if "</think>" not in text: return []                                  # thinking ran out of tokens: no answer
    ans = text.split("</think>")[-1]
    names = {f"{TABLES[j][0]}.{TABLES[j][1]}".lower(): j for j in cands}
    bare = {}
    for j in cands: bare.setdefault(TABLES[j][1].lower(), []).append(j)
    found = re.findall(r'"([^"]+)"', ans.split("[")[-1] if "[" in ans else ans)
    out = []
    for n in found:
        n = n.strip().strip("`").lower()
        j = names.get(n) if n in names else (bare[n][0] if len(bare.get(n, [])) == 1 else None)
        if j is not None and j not in out: out.append(j)
    return out[:K_FINAL]


PICK, FULL15, ROUTE, n_fail = [], [], [], 0
for i, r in enumerate(picks_raw):
    p = parse_pick(r["text"], RER[i][:K_RERANK])
    n_fail += not p
    db = TABLES[p[0]][0] if p else TABLES[RER[i][0]][0]
    p = [j for j in p if TABLES[j][0] == db] or [j for j in RER[i] if TABLES[j][0] == db][:K_FINAL]
    PICK.append(p); ROUTE.append(db)
    FULL15.append(p + [j for j in RER[i] if j not in p][:K_FINAL - len(p)])        # filled to 15 in reranker order, for recall@15
R[f"Qwen3-8B pick, filled to {K_FINAL}"] = rec(FULL15)
R["Qwen3-8B pick, exactly what the generator sees"] = rec(PICK)
routing = 100 * np.mean([d == q["db_id"] for d, q in zip(ROUTE, DEVQ)])

def report():
    """Print and save the report (called before generation too, so a generation failure never loses the retrieval numbers)."""
    L = [f"Paper-2 pilot | BIRD dev, {n} questions | pooled catalog {len(TABLES)} tables in {len(TBL_OF)} databases (BIRD train + dev)",
         f"decider: {used}, fine-tuned on {len(FIT)} train questions (loss: {'BCE + listwise' if LOSS == 'mix' else 'BCE'}) | reranker fitted on {len(VAL)} questions from {len(VAL_DBS)} held-out train databases",
         "", f"{'stage':52s} {'recall':>7s} {'complete':>9s}"]
    L += [f"{k:52s} {r:7.1f} {c:9.1f}" for k, (r, c) in R.items()]
    L += [f"gold tables found by only one list: hybrid only {only_h}, decider only {only_d}",
          f"picker: {n_fail} answers could not be parsed (fell back to the reranker) | tables given to the generator: {np.mean([len(p) for p in PICK]):.1f} on average",
          f"routing accuracy (database of the picked tables): {routing:.1f}%", ""]
    L += ([f"execution accuracy, full pipeline:            {100 * np.mean(ex_pipe):5.1f}",
           f"execution accuracy, Gemma with gold tables:   {100 * np.mean(ex_gold):5.1f}   (generator ceiling)"] if ex_pipe is not None
          else ["execution accuracy: not run yet" + ("" if RUN_GENERATION else " (RUN_GENERATION = False)")])
    L += ["", "minutes: " + " | ".join(f"{k} {v / 60:.1f}" for k, v in TIMES.items()),
          "", "For reference (published): RASL (Claude 3.5 Sonnet-v2) on BIRD dev: recall@15 97.8, EX 53.5 (54.0 with gold tables)."]
    (OUT / f"PILOT_{TAG}.txt").write_text("\n".join(L) + "\n")
    print("\n" + "\n".join(L))


ex_pipe = ex_gold = None
n = len(DEVQ)
report()
if RUN_GENERATION:
    # ---------------------------------------------------------------- 9  Gemma-4-31B-it writes the SQL (3 nearest train examples as few-shot)
    say(f"step 8/9: {GENERATOR} writes the SQL (loading takes 4 to 6 minutes)")
    t0 = time.time()
    DDL = {}
    for db in DEV_DBS:
        con = sqlite3.connect(dbpath(db))
        for name, sql in con.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
            DDL[(db, name.lower())] = sql.strip()
        con.close()
    ddl = lambda j: DDL.get((TABLES[j][0], TABLES[j][1].lower()), BLOCK[j])
    QE_dev = H["dev"][3]
    GEN_PROMPT = """Here are examples of questions and their SQL from other databases:

    {shots}

    Database schema (SQLite):

    {schema}

    Question: {q}
    Evidence: {e}

    Write one SQLite query that answers the question. Return only the query in a ```sql code block."""


    def gen_msgs(table_sets):
        out = []
        for i, (q, ts) in enumerate(zip(DEVQ, table_sets)):
            shots = "\n\n".join(f"Question: {POOL[k]['question']}\nEvidence: {POOL[k].get('evidence', '') or 'none'}\nSQL: {POOL[k]['SQL']}"
                                for k in np.argsort(-(PE @ QE_dev[i]))[:3])
            out.append([{"role": "system", "content": "You are an expert data analyst who writes correct SQLite queries."},
                        {"role": "user", "content": GEN_PROMPT.format(shots=shots, schema="\n\n".join(ddl(j) for j in ts), q=q["question"],
                                                                      e=q.get("evidence", "") or "none")}])
        return out


    GOLD_SETS = [gold_ids(q) for q in DEVQ]
    both = gen_msgs(PICK) + gen_msgs(GOLD_SETS)
    # FP8 weights first (about 31 GB instead of 62 GB; on an A100 vLLM runs them weight-only), text-only, then fallbacks
    gen = llm_run("generator", GENERATOR, both, {"enable_thinking": False}, 8192, 1024, 0.0, 1.0, -1, 0.90,
                  attempts=({"quantization": "fp8", "limit_mm_per_prompt": {"image": 0}}, {"quantization": "fp8"}, {}))
    TIMES["generator"] = time.time() - t0

    say("step 9/9: executing against sqlite")
    from func_timeout import func_timeout, FunctionTimedOut


    def extract(text):
        blocks = re.findall(r"```sql\s*(.*?)```", text, re.S)
        return (blocks[-1] if blocks else text.split("<channel|>")[-1]).strip()


    def run_sql(db, sql):
        con = sqlite3.connect(dbpath(db)); con.text_factory = lambda b: b.decode(errors="ignore")
        try: return con.execute(sql).fetchall()
        finally: con.close()


    def correct(db, sql, q):
        if db not in DEV_DBS: return 0.0                                      # routed to a train database: wrong by definition
        try:
            return float(set(func_timeout(30, run_sql, args=(db, sql))) == set(func_timeout(30, run_sql, args=(q["db_id"], q["SQL"]))))
        except (FunctionTimedOut, Exception):
            return 0.0


    n = len(DEVQ)
    ex_pipe = [correct(ROUTE[i], extract(gen[i]["text"]), q) for i, q in enumerate(DEVQ)]
    ex_gold = [correct(q["db_id"], extract(gen[n + i]["text"]), q) for i, q in enumerate(DEVQ)]
    with open(OUT / f"dev_rows_{TAG}.jsonl", "w") as f:
        for i, q in enumerate(DEVQ):
            f.write(json.dumps(dict(question_id=q.get("question_id"), gold_db=q["db_id"], route=ROUTE[i], gold=[TABLES[j][1] for j in GOLD_SETS[i]],
                                    picked=[f"{TABLES[j][0]}.{TABLES[j][1]}" for j in PICK[i]], sql=extract(gen[i]["text"]), ok=ex_pipe[i],
                                    ok_gold_tables=ex_gold[i], difficulty=q.get("difficulty", ""))) + "\n")


    report()
