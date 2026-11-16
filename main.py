# ============ ONE CELL: final frozen run for paper 2. No LLM, no fine-tuned model. BIRD, Spider, Fiben, BEAVER. A100 40GB is fine. ============
# Frozen method (chosen by the pre-set rules in the selection run, nothing tuned on any test set):
#   candidates  top 100 tables by reciprocal rank fusion of BM25, Qwen3-Embedding-0.6B and BGE-large (all off the shelf).
#               The selection rule tied Qwen3 + BGE (98.4) with BM25 + Qwen3 + BGE (98.3) on BIRD train complete@100; the tie is
#               broken in favour of combining lexical and dense retrieval, decided before any test set was scored.
#   scorers     BM25 (full card), BM25 (name), Qwen3 (full card), Qwen3 (name), BGE (full card), the fused pool score
#   features    120 numbers per (question, table): scores, agreement, database, lexical, coverage, evidence, cell values, types,
#               table metadata, join graph, question context, compressed embedding interaction
#   fuser       LightGBM lambdarank, fitted ONCE on BIRD train questions (gold tables from the gold SQL, parsed with sqlglot)
#   no decider, no LLM, nothing fitted on Spider, Fiben or BEAVER
# Three fusers, all fitted on BIRD train only, differing only in which inputs exist:
#   F_ev      question + BIRD evidence, all 120 features                   -> BIRD dev (RASL's setting)
#   F_noev    question only (no evidence features)                         -> BIRD dev without evidence, Spider dev
#   F_schema  question only, no cell values, column types or row counts    -> Fiben, BEAVER (schema-only catalogs)
# About 1.5 hours the first time (features for ~11,000 BIRD questions, twice); every stage is cached in OUT_DIR.
OUT_DIR = "/content/drive/MyDrive/nift_rasl/final"
OOF_DIR = "/content/drive/MyDrive/nift_rasl/oof"                     # its BIRD value index is reused if present
BEAVER_DIR = "/content/drive/MyDrive/beaver_v3"
KEYS_FILE = "/content/drive/MyDrive/nift_v2/schema_keys.json"        # BEAVER declared foreign keys (optional)
K_CAND = 100
EMB1, EMB2 = "Qwen/Qwen3-Embedding-0.6B", "BAAI/bge-large-en-v1.5"
RUN_EX = False            # appendix: execution accuracy with Gemma writing the SQL from our retrieved tables (done; generations saved)
RUN_ANALYSIS = True       # review analyses from saved outputs: frozen-config ablations (CV on BIRD train), no-cell-value variants, recall@k,
                          # Fiben pool recall, database-clustered bootstrap, EX on correctly routed questions (about 40 minutes, CPU)
EX_SETS = ("bird", "spider")
EX_N = 0                  # questions per dataset for the EX run (0 = all)
# =====================================================================================================================
import os, subprocess, sys, time
T0 = time.time()
def say(msg): print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)
say("step 1/8: installing (1 to 2 minutes)")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rank_bm25", "sentence-transformers", "lightgbm", "remotezip", "rapidfuzz",
                "sqlglot", "pandas", "pyarrow", "huggingface_hub", "func_timeout"], check=True)
if RUN_EX:
    say("  installing vLLM for the EX appendix (3 to 5 minutes, quiet)")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)
from google.colab import drive
drive.mount("/content/drive")
import json, re, zipfile, random, glob, pickle, sqlite3, csv, shutil
from pathlib import Path
import numpy as np, torch, gc
import warnings; warnings.filterwarnings("ignore")
OUT = Path(OUT_DIR); OUT.mkdir(parents=True, exist_ok=True)
TIMES = {}

# ---------------------------------------------------------------- 2  data: BIRD, Spider, Fiben, BEAVER
say("step 2/8: data")
BIRD = "https://bird-bench.oss-cn-beijing.aliyuncs.com/"
DEV, TRN = Path("/content/bird_dev"), Path("/content/bird_train"); DEV.mkdir(exist_ok=True); TRN.mkdir(exist_ok=True)
if not (DEV / "dev.json").exists() or not glob.glob(str(DEV / "**/dev_databases"), recursive=True):
    subprocess.run(["wget", "-q", "--show-progress", "--progress=bar:force:noscroll", "-O", "/content/dev.zip", BIRD + "dev.zip"], check=True)
    with zipfile.ZipFile("/content/dev.zip") as z: z.extractall("/content/bird_raw")
    root = Path(glob.glob("/content/bird_raw/**/dev.json", recursive=True)[0]).parent
    for f in ("dev.json", "dev_tables.json"): (DEV / f).write_bytes((root / f).read_bytes())
    inner = glob.glob(str(root / "dev_databases*.zip"))
    if inner:
        with zipfile.ZipFile(inner[0]) as z: z.extractall(DEV)
    else:
        subprocess.run(["cp", "-r", str(root / "dev_databases"), str(DEV)], check=True)
BIRD_VALS = Path(OOF_DIR) / "values_index.pkl"
need_trn_db = not BIRD_VALS.exists() and not glob.glob(str(TRN / "**/train_databases"), recursive=True)
if need_trn_db:
    say("  downloading BIRD train with its databases (several GB, once)")
    subprocess.run(["wget", "-q", "--show-progress", "--progress=bar:force:noscroll", "-O", "/content/train.zip", BIRD + "train.zip"], check=True)
    with zipfile.ZipFile("/content/train.zip") as z:
        real = [n for n in z.namelist() if "__MACOSX" not in n and not Path(n).name.startswith("._")]
        for n in real:
            if Path(n).name in ("train.json", "train_tables.json"): (TRN / Path(n).name).write_bytes(z.read(n))
        inner = [n for n in real if Path(n).name == "train_databases.zip"]
        z.extract(inner[0], "/content/trn_raw")
    with zipfile.ZipFile(glob.glob("/content/trn_raw/**/train_databases.zip", recursive=True)[0]) as z: z.extractall(TRN)
    os.remove("/content/train.zip")
elif not (TRN / "train.json").exists():
    from remotezip import RemoteZip
    with RemoteZip(BIRD + "train.zip") as z:
        for want in ("train.json", "train_tables.json"):
            (TRN / want).write_bytes(z.read([n for n in z.namelist() if n.endswith("/" + want) or n == want][0]))
SPROOT = Path("/content/spider")
def find_spider():
    for base, dirs, files in os.walk(SPROOT):
        if {"tables.json", "dev.json"} <= set(files) and glob.glob(os.path.join(base, "database", "*", "*.sqlite")): return Path(base)
SPD = find_spider()
if not SPD:
    from huggingface_hub import list_repo_files, hf_hub_download
    zips = [f for f in list_repo_files("HAL-9001/spider-databases", repo_type="dataset") if f.endswith(".zip")]
    zp = hf_hub_download("HAL-9001/spider-databases", zips[0], repo_type="dataset")
    zipfile.ZipFile(zp).extractall(SPROOT)
    for inner in glob.glob(f"{SPROOT}/**/*.zip", recursive=True): zipfile.ZipFile(inner).extractall(os.path.dirname(inner))
    SPD = find_spider()
FD = Path("/content/fiben-benchmark")
if not FD.exists(): subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/IBM/fiben-benchmark.git", str(FD)], check=True)

# ---------------------------------------------------------------- 3  catalogs (one generic structure for every dataset)
camel = lambda s: re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
STOP = set("the a an of in on for to and or is are was were be by with what which who whom whose how many much list give show find "
           "name names all each per than that this these those there their its it as at from among between please tell me do does did "
           "have has had more most less least number state indicate provide calculate identify".split())
def toks(s):
    out = []
    for w in re.findall(r"[a-z0-9]+", camel(s).lower().replace("_", " ")):
        if w in STOP: continue
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out
words = lambda s: re.findall(r"[a-z0-9]+", camel(s).lower().replace("_", " "))
NUMT = ("int", "real", "num", "float", "double", "decimal")


def make_catalog(name, entries, edges):
    """entries: dict(db, table, cols=[(name, natural name, type)], n_pk, fk_cols); edges: (table index, table index) join pairs."""
    C = dict(name=name, TABLES=[(e["db"], e["table"]) for e in entries], COLS=[[(c, nat) for c, nat, _ in e["cols"]] for e in entries],
             CTYPE=[[ty.lower() for *_, ty in e["cols"]] for e in entries], NPK=[e["n_pk"] for e in entries], NFKC=[e["fk_cols"] for e in entries])
    C["NT"] = NT = len(entries); C["NCOL"] = [len(c) for c in C["COLS"]]
    C["CARD"] = [f"{t} (database {d.replace('_', ' ')}): " + ", ".join(c if nat.lower().replace(" ", "") == c.lower().replace("_", "") else f"{c} ({nat})"
                                                                     for c, nat in C["COLS"][k]) for k, (d, t) in enumerate(C["TABLES"])]
    C["NAMECARD"] = [f"{camel(t).replace('_', ' ')} (database {d.replace('_', ' ')})" for d, t in C["TABLES"]]
    C["ADJ"] = {k: set() for k in range(NT)}
    for a, b in edges:
        if a != b: C["ADJ"][a].add(b); C["ADJ"][b].add(a)
    C["TBL_OF"] = {}
    for k, (d, t) in enumerate(C["TABLES"]): C["TBL_OF"].setdefault(d, []).append(k)
    C["TTOK"] = [set(toks(t)) for _, t in C["TABLES"]]
    C["CTOK"] = [set(toks(t) + [w for c, nat in C["COLS"][k] for w in toks(c) + toks(nat)]) for k, (_, t) in enumerate(C["TABLES"])]
    C["CNAMES"] = [{c.lower() for c, _ in C["COLS"][k]} - {"id", "name", "type", "date", "year", "description", "code"} for k in range(NT)]
    DF = {}
    for ts in C["CTOK"]:
        for w in ts: DF[w] = DF.get(w, 0) + 1
    C["IDF"] = {w: np.log(NT / (1 + d)) for w, d in DF.items()}; C["VOCAB"] = sorted(DF); C["VID"] = {w: k for k, w in enumerate(C["VOCAB"])}
    PR = np.zeros(NT)
    for d, ids in C["TBL_OF"].items():
        n = len(ids); loc = {j: k for k, j in enumerate(ids)}; r = np.ones(n) / n
        for _ in range(30):
            nr = np.full(n, 0.15 / n)
            for j in ids:
                nb = C["ADJ"][j]
                for b in nb: nr[loc[b]] += 0.85 * r[loc[j]] / len(nb)
                if not nb: nr += 0.85 * r[loc[j]] / n
            r = nr
        for j in ids: PR[j] = r[loc[j]] * n
    C["PR"] = PR
    C["VALS"], C["NUMR"], C["NROWS"], C["VCOLS"], C["NCOLS_R"] = {}, {}, {}, {}, {}
    return C


def from_tables_json(name, schemas):
    entries, edges = [], []
    for s in schemas:
        cid = {i: (t, c) for i, (t, c) in enumerate(s["column_names_original"])}
        pk = set()
        for p in s["primary_keys"]: pk |= set(p) if isinstance(p, list) else {p}
        fkc = {a for a, b in s["foreign_keys"]} | {b for a, b in s["foreign_keys"]}
        base = len(entries)
        for ti, t in enumerate(s["table_names_original"]):
            mine = [(i, c, nat, ty) for i, ((tt, c), (_, nat), ty) in enumerate(zip(s["column_names_original"], s["column_names"], s["column_types"])) if tt == ti]
            entries.append(dict(db=s["db_id"], table=t, cols=[(c, nat, ty) for _, c, nat, ty in mine], n_pk=sum(i in pk for i, *_ in mine),
                                fk_cols=sum(i in fkc for i, *_ in mine)))
        edges += [(base + cid[a][0], base + cid[b][0]) for a, b in s["foreign_keys"]]
    return make_catalog(name, entries, edges)


def add_values(C, dbfile, cache):
    """Distinct text values (up to 20,000 per column), numeric ranges and row counts, read from the sqlite files."""
    if cache.exists():
        C["VALS"], C["NUMR"], C["NROWS"] = pickle.loads(cache.read_bytes())
    else:
        for d, ids in C["TBL_OF"].items():
            if d not in dbfile: continue
            con = sqlite3.connect(f"file:{dbfile[d]}?mode=ro", uri=True); con.text_factory = lambda b: b.decode(errors="ignore")
            deadline = [0]; con.set_progress_handler(lambda: 1 if time.time() > deadline[0] else 0, 20000)
            for j in ids:
                t = C["TABLES"][j][1]
                try:
                    deadline[0] = time.time() + 5; C["NROWS"][j] = max(0, con.execute(f'SELECT max(rowid) FROM "{t}"').fetchone()[0] or 0)
                except Exception:
                    C["NROWS"][j] = 0
                for (c, _), ty in zip(C["COLS"][j], C["CTYPE"][j]):
                    try:
                        deadline[0] = time.time() + 5
                        if any(k in ty for k in NUMT):
                            lo, hi = con.execute(f'SELECT min("{c}"), max("{c}") FROM "{t}"').fetchone()
                            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)): C["NUMR"][(j, c)] = (lo, hi)
                        else:
                            vs = con.execute(f'SELECT DISTINCT "{c}" FROM "{t}" WHERE "{c}" IS NOT NULL LIMIT 20000').fetchall()
                            C["VALS"][(j, c)] = {str(v[0]).strip().lower() for v in vs if v[0] is not None and 1 < len(str(v[0]).strip()) <= 60}
                    except Exception:
                        pass
            con.close()
        cache.write_bytes(pickle.dumps((C["VALS"], C["NUMR"], C["NROWS"])))
    C["VCOLS"], C["NCOLS_R"] = {}, {}
    for (j, c) in C["VALS"]: C["VCOLS"].setdefault(j, []).append(c)
    for (j, c) in C["NUMR"]: C["NCOLS_R"].setdefault(j, []).append(c)


import sqlglot
from sqlglot import exp
def sql_gold(C, sql, db, dialect):
    """Gold tables from the gold SQL, parsed with sqlglot (CTE names removed). None if it does not parse."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
        names = {t.name.lower() for t in tree.find_all(exp.Table)} - {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    except Exception:
        return None
    return {k for k in C["TBL_OF"].get(db, []) if C["TABLES"][k][1].lower() in names}


say("step 3/8: catalogs and gold tables")
# BIRD: train + dev databases pooled (RASL's catalog), same order as the earlier runs so the cached value index lines up
CB = from_tables_json("BIRD", json.loads((TRN / "train_tables.json").read_text()) + json.loads((DEV / "dev_tables.json").read_text()))
dbf = {}
for b in glob.glob(str(DEV / "**/dev_databases"), recursive=True) + glob.glob(str(TRN / "**/train_databases"), recursive=True):
    for f in glob.glob(b + "/*/*.sqlite"): dbf[Path(f).parent.name] = f
add_values(CB, dbf, BIRD_VALS if BIRD_VALS.exists() else OUT / "values_bird.pkl")
def bird_q(rows):
    out = []
    for q in rows:
        g = sql_gold(CB, q["SQL"], q["db_id"], "sqlite")
        if g: out.append(dict(question=q["question"], evidence=q.get("evidence", "") or "", gold=g, db=q["db_id"], qid=q.get("question_id"), sql=q["SQL"],
                              difficulty=q.get("difficulty", "")))
    return out
BQ_TR = bird_q(json.loads((TRN / "train.json").read_text())); BQ_DEV = bird_q(json.loads((DEV / "dev.json").read_text()))
# Spider: train + dev databases pooled (RASL's catalog: 166 databases, 876 tables), dev questions
CS = from_tables_json("Spider", json.loads((SPD / "tables.json").read_text()))
add_values(CS, {Path(f).parent.name: f for f in glob.glob(str(SPD / "database/*/*.sqlite"))}, OUT / "values_spider.pkl")
SQ = []
for q in json.loads((SPD / "dev.json").read_text()):
    g = sql_gold(CS, q["query"], q["db_id"], "sqlite")
    if g: SQ.append(dict(question=q["question"], evidence="", gold=g, db=q["db_id"], sql=q["query"]))
# Fiben: one database, schema from the published DDL, gold from the PostgreSQL gold SQL
ddl = (FD / "FIBEN.sql").read_text(); fe, TI = [], {}
for m in re.finditer(r"CREATE TABLE\s+(\w+)\s*\((.*?)\);", ddl, re.S):
    parts = [p.strip() for p in re.split(r",\s*(?![^()]*\))", m.group(2)) if p.strip() and not p.strip().upper().startswith("PRIMARY KEY")]
    TI[m.group(1).upper()] = len(fe)
    fe.append(dict(db="fiben", table=m.group(1).upper(), cols=[(p.split()[0], p.split()[0].replace("_", " ").lower(), (p.split() + ["text"])[1]) for p in parts],
                   n_pk=0, fk_cols=0))
fedges = [(TI[a.upper()], TI[c.upper()]) for a, b, c, d in re.findall(
    r"ALTER TABLE\s+(\w+)\s+ADD CONSTRAINT\s+\w+\s+FOREIGN KEY\s*\((\w+)\)\s*REFERENCES\s+(\w+)\s*\((\w+)\)", ddl) if a.upper() in TI and c.upper() in TI]
CF = make_catalog("Fiben", fe, fedges)
FQ = []
for q in json.loads((FD / "FIBEN_Queries.json").read_text()):
    g = sql_gold(CF, q["SQL"], "fiben", "postgres")
    if g: FQ.append(dict(question=q["question"].strip(), evidence="", gold=g))
# BEAVER: current release, three warehouses pooled, a sample of synthetic questions and all real ones
import pandas as pd
findf = lambda name: (glob.glob(f"{BEAVER_DIR}/**/{name}", recursive=True) or [None])[0]
CV_, BVQ = None, []
if findf("table_dw.parquet"):
    KEYS = json.loads(Path(KEYS_FILE).read_text()) if Path(KEYS_FILE).exists() else {}
    be = []
    for db in ("dw", "nova", "neutron"):
        for _, r in pd.read_parquet(findf(f"table_{db}.parquet")).iterrows():
            cs = r["column_names"]; cs = json.loads(cs) if isinstance(cs, str) else list(cs)
            be.append(dict(db=db, table=str(r["table_name"]), cols=[(str(c), str(c).replace("_", " ").lower(), "text") for c in cs], n_pk=0, fk_cols=0))
    IDX = {(e["db"], e["table"].upper()): k for k, e in enumerate(be)}
    bedges = []
    for k, e in enumerate(be):
        for a, ref, b in KEYS.get(e["db"], {}).get(e["table"].upper(), {}).get("fk", []):
            o = IDX.get((e["db"], str(ref).upper()))
            if o is not None: bedges.append((k, o)); be[k]["fk_cols"] += len(a)
    CV_ = make_catalog("BEAVER", be, bedges)
    for sp in ("dw", "nova", "neutron", "dw_real"):
        f_ = findf(f"query_{sp}.parquet")
        if not f_: continue
        for _, r in pd.read_parquet(f_).iterrows():
            tabs = r["tables"]; tabs = json.loads(tabs) if isinstance(tabs, str) else list(tabs)
            g = {IDX.get((str(r["db"]), str(x).upper())) for x in tabs}
            if tabs and None not in g: BVQ.append(dict(question=str(r["question"]), evidence="", gold=g, real=sp == "dw_real"))
    syn = [q for q in BVQ if not q["real"]]
    BVQ = random.Random(7).sample(syn, min(1000, len(syn))) + [q for q in BVQ if q["real"]]
print(f"BIRD {CB['NT']} tables, {len(BQ_TR)} train / {len(BQ_DEV)} dev questions | Spider {CS['NT']} tables, {len(SQ)} dev questions | "
      f"Fiben {CF['NT']} tables, {len(FQ)} questions | BEAVER {CV_['NT'] if CV_ else 0} tables, {len(BVQ)} questions")
print("RASL catalogs: BIRD 597 tables / 1,534 questions | Spider 876 / 1,034 | Fiben 152 / 279")

# ---------------------------------------------------------------- 4  frozen scorers
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
EMBS = {}
def embedder(k):
    if k not in EMBS: EMBS[k] = SentenceTransformer(EMB1 if k == "e1" else EMB2, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    return EMBS[k]
rankm = lambda X: np.argsort(np.argsort(-X, 1), 1)
qtext = lambda q, ev: (q["question"] + (" " + q["evidence"] if ev and q["evidence"] else "")).strip()


def scorers(C, Q, ev, tag):
    f = OUT / f"scores_{tag}_pool3.npz"
    if f.exists():
        z = np.load(f); return {k: z[k] for k in z.files}
    T_ = [qtext(q, ev) for q in Q]; S = {}
    for nm, docs in (("bm25_full", C["CARD"]), ("bm25_name", C["NAMECARD"])):
        b = BM25Okapi([words(c) or ["_"] for c in docs]); S[nm] = np.stack([b.get_scores(words(t)) for t in T_]).astype(np.float32)
    e1, e2 = embedder("e1"), embedder("e2")
    QE = e1.encode(T_, prompt_name="query", normalize_embeddings=True, batch_size=64)
    TE = e1.encode(C["CARD"], normalize_embeddings=True, batch_size=16)
    S["dense_e1"] = (QE @ TE.T).astype(np.float32); S["dense_e1_name"] = (QE @ e1.encode(C["NAMECARD"], normalize_embeddings=True).T).astype(np.float32)
    Q2 = e2.encode(T_, prompt="Represent this sentence for searching relevant passages: ", normalize_embeddings=True, batch_size=64)
    S["dense_e2"] = (Q2 @ e2.encode(C["CARD"], normalize_embeddings=True, batch_size=16).T).astype(np.float32)
    S["qe"], S["te"] = QE.astype(np.float16), TE.astype(np.float16)
    S["pool"] = (1 / (60 + rankm(S["bm25_full"])) + 1 / (60 + rankm(S["dense_e1"])) + 1 / (60 + rankm(S["dense_e2"]))).astype(np.float32)
    np.savez(f, **S); return S


# ---------------------------------------------------------------- 5  features (the public question view and the catalog only)
from rapidfuzz.process import cdist
from rapidfuzz.distance import Indel
SCN = {"pool": "pool", "bm25_full": "bm25_full", "bm25_name": "bm25_name", "dense_e1": "dense_e1", "dense_e1_name": "dense_e1_name", "dense_e2": "dense_e2"}
AGG = re.compile(r"\b(how many|number of|count|average|mean|total|sum|highest|lowest|most|least|maximum|minimum|top|largest|smallest|oldest|youngest|best|worst)\b")
CMP = re.compile(r"(more than|less than|greater|fewer|over|under|at least|at most|above|below|>|<|between)")
PCT = re.compile(r"\b(percent|percentage|ratio|rate|proportion|times)\b")
TIMEW = re.compile(r"\b(year|date|when|month|day|born|since|before|after|age|(19|20)\d\d)\b")
CAPS_STOP = {"what", "how", "list", "which", "please", "among", "give", "name", "calculate", "state", "indicate", "provide", "find", "identify",
             "in", "for", "the", "who", "when", "where", "is", "are", "of", "on", "a", "an", "and", "or", "by", "with", "from", "to", "at", "as", "if"}


def literals(question, evidence):
    text = question + " " + evidence
    lit = {a or b for a, b in re.findall(r"'([^']{2,60})'|\"([^\"]{2,60})\"", text)}
    for span in re.findall(r"\b([A-Z][\w\-\.']*(?:\s+[A-Z][\w\-\.']*)*)", question):
        ws = [w for w in span.split() if w.lower() not in CAPS_STOP]
        if ws: lit.add(" ".join(ws))
    return {l.strip().lower() for l in lit if len(l.strip()) > 1}, [float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])", text)]


def bfs(A, src, dst):
    if src == dst: return 0
    seen, fr = {src}, [src]
    for d in range(1, 6):
        nx = []
        for a in fr:
            for b in A[a]:
                if b == dst: return d
                if b not in seen: seen.add(b); nx.append(b)
        fr = nx
    return 9


def feats(C, S, i, question, evidence):
    """120 features for each of question i's candidates. Inputs: question text, evidence text, catalog, frozen scores. No labels."""
    cand = np.argsort(-S["pool"][i])[:K_CAND]
    sc = {k: S[v][i, cand] for k, v in SCN.items()}
    R = {k: np.argsort(np.argsort(-v)) for k, v in sc.items()}; Z = {k: (v - v.mean()) / (v.std() + 1e-6) for k, v in sc.items()}
    TOP = {k: v.max() for k, v in sc.items()}
    dbs = [C["TABLES"][j][0] for j in cand]; db_stats = {}
    for k in ("pool", "dense_e1"):
        sm = np.exp(Z[k] - Z[k].max()); sm /= sm.sum(); mx, mean, mass, n = {}, {}, {}, {}
        for c, d in enumerate(dbs):
            mx[d] = max(mx.get(d, -1e9), sc[k][c]); mean[d] = mean.get(d, 0) + sc[k][c]; mass[d] = mass.get(d, 0) + sm[c]; n[d] = n.get(d, 0) + 1
        rk = {d: r for r, d in enumerate(sorted(mx, key=lambda d: -mx[d]))}; order = {}
        for c in np.argsort(-sc[k]): order.setdefault(dbs[c], []).append(c)
        rin = {c: r for d in order for r, c in enumerate(order[d])}
        db_stats[k] = (mx, {d: mean[d] / n[d] for d in n}, mass, rk, rin, n)
    top1 = cand[np.argmax(sc["pool"])]; top5 = set(cand[np.argsort(-sc["pool"])[:5]]); top3 = list(cand[np.argsort(-sc["pool"])[:3]])
    ql, evl = question.lower(), evidence.lower()
    qtall = set(toks(question)) | set(toks(evidence)); qw = [w for w in qtall if len(w) >= 4]
    best_tok = cdist(qw, C["VOCAB"], scorer=Indel.normalized_similarity, workers=-1).max(0) if qw else np.zeros(len(C["VOCAB"]))
    lits, nums = literals(question, evidence)
    fl = dict(agg=float(bool(AGG.search(ql))), cmp=float(bool(CMP.search(ql + " " + evl))), pct=float(bool(PCT.search(ql + " " + evl))), time=float(bool(TIMEW.search(ql))))
    hs = np.exp(Z["pool"] - Z["pool"].max()); hs /= hs.sum(); srt = np.sort(sc["pool"])
    qctx = {"q:n_words": len(ql.split()), "q:n_content_tokens": len(qtall), "q:n_literals": len(lits), "q:n_numbers": len(nums),
            "q:has_evidence": float(bool(evidence)), "q:top_entropy": float(-(hs * np.log(hs + 1e-12)).sum()), "q:top_gap": float(srt[-1] - srt[-2]),
            **{f"q:{k}": v for k, v in fl.items()}}
    vmatch = {c: {l for col in C["VCOLS"].get(j, []) for l in lits if l in C["VALS"][(j, col)]} for c, j in enumerate(cand)}
    lit_db = {}
    for c, d in enumerate(dbs):
        for l in vmatch[c]: lit_db.setdefault((d, l), set()).add(c)
    cover1 = C["CTOK"][top1]; cover3 = set().union(*[C["CTOK"][j] for j in top3]); A = C["ADJ"]; rows = []
    for c, j in enumerate(cand):
        d = dbs[c]; f = {}
        for k in sc:
            f[f"score:{k}"] = sc[k][c]; f[f"score:{k}_rank"] = R[k][c]; f[f"score:{k}_z"] = Z[k][c]
            f[f"score:{k}_gap_best"] = sc[k][c] - TOP[k]; f[f"score:{k}_pctl"] = R[k][c] / len(cand)
        rk_ = np.array([R[k][c] for k in sc])
        f.update({"agree:rank_min": rk_.min(), "agree:rank_max": rk_.max(), "agree:rank_mean": rk_.mean(), "agree:rank_std": rk_.std(),
                  "agree:n_top5": float((rk_ < 5).sum()), "agree:n_top15": float((rk_ < 15).sum()), "agree:dense_vs_bm25": R["dense_e1"][c] - R["bm25_full"][c],
                  "agree:e1_vs_e2": R["dense_e1"][c] - R["dense_e2"][c], "agree:name_vs_full": R["dense_e1_name"][c] - R["dense_e1"][c]})
        for k in ("pool", "dense_e1"):
            mx, mean, mass, rkd, rin, n = db_stats[k]
            f.update({f"db:{k}_max": mx[d], f"db:{k}_mean": mean[d], f"db:{k}_mass": mass[d], f"db:{k}_rank": rkd[d], f"db:{k}_rank_in_db": rin[c],
                      f"db:{k}_minus_dbmax": sc[k][c] - mx[d]})
        f.update({"db:n_candidates": db_stats["pool"][5][d], "db:n_tables": len(C["TBL_OF"][d])})
        ct = C["CTOK"][j]; ids = [C["VID"][w] for w in ct]
        f.update({"lex:tname_overlap": len(qtall & C["TTOK"][j]) / max(1, len(C["TTOK"][j])), "lex:col_overlap": len(qtall & ct) / max(1, len(ct)),
                  "lex:idf_overlap": sum(C["IDF"][w] for w in qtall & ct), "lex:n_overlap": len(qtall & ct),
                  "lex:fuzzy_max": max((best_tok[x] for x in ids), default=0), "lex:fuzzy_n_over_0.8": float(sum(best_tok[x] > 0.8 for x in ids)),
                  "lex:col_fullname_in_q": float(sum(1 for _, nat in C["COLS"][j] if len(nat) > 3 and nat.lower() in ql)),
                  "lex:tname_in_q": float(C["TABLES"][j][1].lower().replace("_", " ") in ql)})
        f.update({"cov:q_tokens_covered": len(qtall & ct) / max(1, len(qtall)), "cov:gain_over_top1": len((qtall & ct) - cover1) / max(1, len(qtall)),
                  "cov:gain_over_top3": len((qtall & ct) - cover3) / max(1, len(qtall))})
        f.update({"ev:table_named": float(bool(re.search(r"(?<![\w])" + re.escape(C["TABLES"][j][1].lower()) + r"(?![\w])", evl))) if evl else 0.0,
                  "ev:n_cols_named": float(sum(1 for cc, _ in C["COLS"][j] if re.search(r"(?<![\w])" + re.escape(cc.lower()) + r"(?![\w])", evl))) if evl else 0.0,
                  "ev:n_natnames_named": float(sum(1 for _, nat in C["COLS"][j] if len(nat) > 3 and nat.lower() in evl)) if evl else 0.0})
        uniq = sum(1 for l in vmatch[c] if lit_db.get((d, l)) == {c})
        inr = sum(1 for col in C["NCOLS_R"].get(j, []) for x in nums if C["NUMR"][(j, col)][0] <= x <= C["NUMR"][(j, col)][1])
        f.update({"val:n_literals_matched": len(vmatch[c]), "val:frac_literals_matched": len(vmatch[c]) / max(1, len(lits)),
                  "val:n_cols_with_match": float(sum(1 for col in C["VCOLS"].get(j, []) if any(l in C["VALS"][(j, col)] for l in lits))),
                  "val:unique_in_db": float(uniq), "val:numbers_in_range": float(inr),
                  "val:frac_numbers_in_range": inr / max(1, len(nums) * max(1, len(C["NCOLS_R"].get(j, []))))})
        ty = C["CTYPE"][j]; nn = sum(any(k in t for k in NUMT) for t in ty)
        nd = sum(("date" in t or "time" in t) for t in ty) + sum(any(w in cc.lower() for w in ("date", "year", "time")) for cc, _ in C["COLS"][j])
        f.update({"type:n_numeric": nn, "type:frac_numeric": nn / max(1, C["NCOL"][j]), "type:n_datelike": nd, "type:cmp_x_numeric": fl["cmp"] * nn,
                  "type:agg_x_numeric": fl["agg"] * nn, "type:time_x_date": fl["time"] * nd, "type:pct_x_numeric": fl["pct"] * nn})
        f.update({"meta:n_cols": C["NCOL"][j], "meta:n_pk": C["NPK"][j], "meta:n_fk_cols": C["NFKC"][j], "meta:log_rows": np.log1p(C["NROWS"].get(j, 0)),
                  "meta:junction": float(C["NFKC"][j] >= 2 and C["NCOL"][j] <= 4), "meta:name_len": len(C["TABLES"][j][1])})
        f.update({"graph:degree": len(A[j]), "graph:pagerank": C["PR"][j], "graph:dist_top1": bfs(A, top1, j), "graph:n_nbr_top5": len(A[j] & top5),
                  "graph:bridge": float(len(A[j] & top5) >= 2), "graph:shared_cols_top1": len(C["CNAMES"][j] & C["CNAMES"][top1]) if j != top1 else 0,
                  "graph:shared_cols_top5": float(sum(len(C["CNAMES"][j] & C["CNAMES"][t]) > 0 for t in top5 if t != j))})
        f.update(qctx); rows.append(f)
    return cand, rows


NAMES = None
def matrix(C, S, Q, ev, tag):
    """Features for every question, cached. Returns X (rows = question x candidate), labels, question index per row, candidate ids."""
    global NAMES
    f = OUT / f"features_{tag}_pool3.pkl"
    if f.exists():
        X, y, g, cj, NAMES = pickle.loads(f.read_bytes()); return X, y, g, cj
    X, y, g, cj = [], [], [], []
    for i, q in enumerate(Q):
        cand, rows = feats(C, S, i, q["question"], q["evidence"] if ev else "")
        if NAMES is None: NAMES = list(rows[0])
        X.append(np.array([[r[k] for k in NAMES] for r in rows], dtype=np.float32)); y += [int(j in q["gold"]) for j in cand]; g += [i] * len(cand); cj.append(cand)
        if i % 2000 == 0: say(f"    {tag}: features {i}/{len(Q)}")
    X = np.nan_to_num(np.vstack(X), nan=0.0, posinf=0.0, neginf=0.0); y, g, cj = np.array(y), np.array(g), np.concatenate(cj)
    f.write_bytes(pickle.dumps((X, y, g, cj, NAMES))); return X, y, g, cj


def with_emb(X, S, g, cj, pca):
    qt = S["qe"][g].astype(np.float32) * S["te"][cj].astype(np.float32)
    return np.hstack([X, pca.transform(qt)])


# ---------------------------------------------------------------- 6  BIRD train: features (with and without evidence) and the three fusers
say("step 4/8: BIRD scorers and features (the slow part: ~11,000 questions, twice)")
t0 = time.time()
from sklearn.decomposition import PCA
from lightgbm import LGBMRanker
SB_ev_tr, SB_ev_dev = scorers(CB, BQ_TR, True, "bird_train_ev"), scorers(CB, BQ_DEV, True, "bird_dev_ev")
SB_no_tr, SB_no_dev = scorers(CB, BQ_TR, False, "bird_train_noev"), scorers(CB, BQ_DEV, False, "bird_dev_noev")
M = {}
for tag, C_, S_, Q_, ev in (("bird_train_ev", CB, SB_ev_tr, BQ_TR, True), ("bird_dev_ev", CB, SB_ev_dev, BQ_DEV, True),
                            ("bird_train_noev", CB, SB_no_tr, BQ_TR, False), ("bird_dev_noev", CB, SB_no_dev, BQ_DEV, False)):
    M[tag] = matrix(C_, S_, Q_, ev, tag)
TIMES["BIRD scorers + features"] = time.time() - t0
X_, y_, g_, c_ = M["bird_train_ev"]
pca = PCA(16, random_state=0).fit((SB_ev_tr["qe"][g_].astype(np.float32) * SB_ev_tr["te"][c_].astype(np.float32))[::9])
ALL = NAMES + [f"emb:pc{k}" for k in range(16)]; col = {n: k for k, n in enumerate(ALL)}
FSETS = {"F_ev": ALL,
         "F_noev": [n for n in ALL if not n.startswith("ev:") and n != "q:has_evidence"]}
FSETS["F_schema"] = [n for n in FSETS["F_noev"] if n.split(":")[0] not in ("val", "type") and n != "meta:log_rows"]
say("step 5/8: fitting the three fusers on BIRD train (lambdarank)")
t0 = time.time()
FUS = {}
for fn, train_tag in (("F_ev", "bird_train_ev"), ("F_noev", "bird_train_noev"), ("F_schema", "bird_train_noev")):
    X_, y_, g_, c_ = M[train_tag]; S_ = SB_ev_tr if train_tag.endswith("_ev") else SB_no_tr
    ix = [col[n] for n in FSETS[fn]]
    FUS[fn] = LGBMRanker(objective="lambdarank", n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, random_state=0, verbose=-1).fit(
        with_emb(X_, S_, g_, c_, pca)[:, ix], y_, group=np.bincount(g_))
    say(f"  {fn}: {len(ix)} features")
TIMES["fusers"] = time.time() - t0

# ---------------------------------------------------------------- 7  evaluation: every test set scored once
say("step 6/8: Spider, Fiben and BEAVER scorers and features")
t0 = time.time()
SETS = [("BIRD dev, with evidence (RASL's setting)", CB, SB_ev_dev, BQ_DEV, True, "bird_dev_ev", "F_ev", ("BIRD",)),
        ("BIRD dev, question only", CB, SB_no_dev, BQ_DEV, False, "bird_dev_noev", "F_noev", ("BIRD",))]
SS = scorers(CS, SQ, False, "spider_dev"); M["spider_dev"] = matrix(CS, SS, SQ, False, "spider_dev")
SETS.append(("Spider dev", CS, SS, SQ, False, "spider_dev", "F_noev", ("Spider",)))
SF = scorers(CF, FQ, False, "fiben"); M["fiben"] = matrix(CF, SF, FQ, False, "fiben")
SETS.append(("Fiben", CF, SF, FQ, False, "fiben", "F_schema", ("Fiben",)))
if CV_:
    SV = scorers(CV_, BVQ, False, "beaver"); M["beaver"] = matrix(CV_, SV, BVQ, False, "beaver")
    SETS.append(("BEAVER, 1,000 synthetic questions", CV_, SV, BVQ, False, "beaver", "F_schema", ("syn",)))
    SETS.append(("BEAVER, real questions", CV_, SV, BVQ, False, "beaver", "F_schema", ("real",)))
TIMES["other scorers + features"] = time.time() - t0
del EMBS; gc.collect(); torch.cuda.empty_cache()
RASL = {"BIRD": "RASL retriever R@5 70.5 R@15 92.6 | RASL full (Claude 3.5 Haiku + Sonnet) R@5 97.5 R@15 97.8 | DBCopilot 85.8 / 94.6",
        "Spider": "RASL retriever R@5 72.1 R@15 94.1 | RASL full R@5 97.0 R@15 98.0 | DBCopilot 91.6 / 97.6",
        "Fiben": "RASL retriever R@5 34.6 R@15 64.5 | RASL full R@5 69.1 R@15 69.2 (279 questions) | DBCopilot 41.1 / 56.9",
        "syn": "", "real": "paper 1 NIFT-SQL (trained on BEAVER synthetic questions): R@15 80.8, C@15 54.5 | its search: R@15 74.3, C@15 42.1"}
rs = np.random.default_rng(0)
def ci(x):
    b = [x[rs.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return f"{100*x.mean():5.1f} [{100*np.percentile(b, 2.5):.1f}, {100*np.percentile(b, 97.5):.1f}]"


def metrics(score, y, g, qsel, ngold, ks=(5, 10, 15)):
    out = {}
    for k in ks:
        hit = np.array([y[g == i][np.argsort(-score[g == i])[:k]].sum() for i in qsel])
        out[k] = (hit / ngold, (hit >= ngold).astype(float))
    return out


say("step 7/8: scoring every test set once")
REP = [f"Final frozen run | {time.strftime('%Y-%m-%d %H:%M')} | no LLM, no fine-tuned model | fusers fitted on BIRD train only ({len(BQ_TR)} questions)",
       f"candidates: top {K_CAND} by RRF of BM25 + Qwen3-Embedding-0.6B + BGE-large | features: {len(ALL)} | gold tables: sqlglot", ""]
PRED = {}
for title, C_, S_, Q_, ev, tag, fn, key in SETS:
    X_, y_, g_, c_ = M[tag]
    qsel = [i for i, q in enumerate(Q_) if key[0] not in ("syn", "real") or q["real"] == (key[0] == "real")]
    ng = np.array([len(Q_[i]["gold"]) for i in qsel])
    Xe = with_emb(X_, S_, g_, c_, pca); sc_f = FUS[fn].predict(Xe[:, [col[n] for n in FSETS[fn]]]); PRED[title] = sc_f
    rows = {"BM25 alone": X_[:, NAMES.index("score:bm25_full")], "BM25 + Qwen3 + BGE pool alone": X_[:, NAMES.index("score:pool")], f"fuser {fn}": sc_f}
    REP += [f"== {title}: {len(qsel)} questions, {C_['NT']} tables, {ng.mean():.2f} gold tables per question, fuser {fn} ({len(FSETS[fn])} features)",
            f"   {'method':26s} {'R@5 [95% CI]':>20s} {'R@10':>6s} {'R@15 [95% CI]':>20s} {'C@5':>6s} {'C@10':>6s} {'C@15':>6s}"]
    for nm, v in rows.items():
        m = metrics(v, y_, g_, qsel, ng)
        REP.append(f"   {nm:26s} {ci(m[5][0]):>20s} {100*m[10][0].mean():6.1f} {ci(m[15][0]):>20s} {100*m[5][1].mean():6.1f} "
                   f"{100*m[10][1].mean():6.1f} {100*m[15][1].mean():6.1f}")
    if RASL[key[0]]: REP.append(f"   published: {RASL[key[0]]}")
    REP.append("")

# ---------------------------------------------------------------- 8  checks: leakage audit (BIRD dev), feature dump, latency
say("step 8/8: leakage audit, feature dump, latency")
from sklearn.metrics import roc_auc_score
X_, y_, g_, c_ = M["bird_dev_ev"]; Xe = with_emb(X_, SB_ev_dev, g_, c_, pca)
auc = {}
for n in ALL:
    v = Xe[:, col[n]]
    if np.std(v) > 0: a = roc_auc_score(y_, v); auc[n] = max(a, 1 - a)
sus = [(n, a) for n, a in auc.items() if a > 0.98]
REP += ["leakage audit, BIRD dev: features that alone separate gold from non-gold with AUC > 0.98: " + (", ".join(f"{n} {a:.3f}" for n, a in sus) or "none"),
        "strongest single features: " + ", ".join(f"{n} {a:.3f}" for n, a in sorted(auc.items(), key=lambda x: -x[1])[:5])]
gain = FUS["F_ev"].booster_.feature_importance("gain"); top = np.argsort(-gain)[:15]
REP += ["top 15 features of F_ev by gain: " + ", ".join(f"{FSETS['F_ev'][k]} {100*gain[k]/gain.sum():.1f}%" for k in top)]
sc_dev = PRED["BIRD dev, with evidence (RASL's setting)"]
with open(OUT / "FEATURE_DUMP_bird_dev.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["question_id", "question", "evidence", "table", "is_gold", "fuser_score", "fuser_rank"] + ALL)
    for i in random.Random(5).sample(range(len(BQ_DEV)), 30):
        ix = np.where(g_ == i)[0]; order = ix[np.argsort(-sc_dev[ix])]
        for rnk, r in enumerate(list(order[:20]) + [r for r in order[20:] if y_[r]]):
            w.writerow([BQ_DEV[i]["qid"], BQ_DEV[i]["question"], BQ_DEV[i]["evidence"], ".".join(CB["TABLES"][c_[r]]), y_[r], round(float(sc_dev[r]), 4),
                        int(np.where(order == r)[0][0]) + 1] + [round(float(x), 4) for x in Xe[r]])
# latency per question, batch size 1, the complete pipeline from question text to ranked tables
e1 = SentenceTransformer(EMB1, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16}); e2 = SentenceTransformer(EMB2, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
T1, T1n, T2 = (e1.encode(CB["CARD"], normalize_embeddings=True), e1.encode(CB["NAMECARD"], normalize_embeddings=True), e2.encode(CB["CARD"], normalize_embeddings=True))
b1, b2 = BM25Okapi([words(c) or ["_"] for c in CB["CARD"]]), BM25Okapi([words(c) or ["_"] for c in CB["NAMECARD"]])
TT = {k: [] for k in ("embed question (2 models)", "BM25 (2 views)", "dense scores + pool", "features", "fuser")}
for i in range(min(105, len(BQ_DEV))):
    q = BQ_DEV[i]; t = time.perf_counter(); tx = qtext(q, True)
    a1 = e1.encode([tx], prompt_name="query", normalize_embeddings=True); a2 = e2.encode([tx], prompt="Represent this sentence for searching relevant passages: ", normalize_embeddings=True)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    s1, s2 = b1.get_scores(words(tx)), b2.get_scores(words(tx)); t2 = time.perf_counter()
    d1, dn, d2 = a1 @ T1.T, a1 @ T1n.T, a2 @ T2.T
    r_ = lambda v: np.argsort(np.argsort(-v)); pool = 1 / (60 + r_(s1)) + 1 / (60 + r_(d1[0])) + 1 / (60 + r_(d2[0])); t3 = time.perf_counter()
    S1 = {"pool": pool[None], "bm25_full": s1[None], "bm25_name": s2[None], "dense_e1": d1, "dense_e1_name": dn, "dense_e2": d2}
    cand, rows = feats(CB, S1, 0, q["question"], q["evidence"])
    xr = np.array([[r[k] for k in NAMES] for r in rows], dtype=np.float32)
    xr = np.nan_to_num(np.hstack([xr, pca.transform(a1.astype(np.float32) * T1[cand].astype(np.float32))])); t4 = time.perf_counter()
    FUS["F_ev"].predict(xr[:, [col[n] for n in FSETS["F_ev"]]]); t5 = time.perf_counter()
    for k, v in zip(TT, (t1 - t, t2 - t1, t3 - t2, t4 - t3, t5 - t4)): TT[k].append(1000 * v)
TT = {k: v[5:] for k, v in TT.items()}; tot = np.sum([np.array(v) for v in TT.values()], 0)
REP += ["", f"latency per question, batch size 1, {torch.cuda.get_device_name(0)}, ms (median / 90th percentile):"]
REP += [f"   {k:28s} {np.median(v):7.1f} {np.percentile(v, 90):7.1f}" for k, v in TT.items()]
REP += [f"   {'total':28s} {np.median(tot):7.1f} {np.percentile(tot, 90):7.1f}   (two embedders on GPU; BM25, features and fuser on CPU; no LLM call)",
        "", f"feature values for 30 BIRD dev questions: {OUT / 'FEATURE_DUMP_bird_dev.csv'}",
        "minutes: " + " | ".join(f"{k} {v / 60:.1f}" for k, v in TIMES.items())]
(OUT / "FINAL.txt").write_text("\n".join(REP) + "\n")
print("\n" + "\n".join(REP))

# ================================================================ APPENDIX: execution accuracy, Gemma writes the SQL from our retrieved tables
# Not part of the method. Our ranked tables (the frozen fuser above) -> the database of the top-ranked table -> Gemma (zero-shot,
# greedy, thinking off) -> SQL -> executed and compared with the gold result (set equality, BIRD's official comparison).
# Conditions: our top 5 and top 15 (tables of the routed database only), and the gold tables (Gemma's own ceiling).
# RASL (Claude 3.5 Sonnet writes the SQL): BIRD N=15 53.5 EX (gold tables 54.0), Spider 64.5 (gold 69.0).
if RUN_EX:
    t_ex = time.time()
    for v in ("e1", "e2"): globals().pop(v, None)
    gc.collect(); torch.cuda.empty_cache()
    from func_timeout import func_timeout, FunctionTimedOut
    SPDB = {Path(f).parent.name: f for f in glob.glob(str(SPD / "database/*/*.sqlite"))}
    DBFILE = {"BIRD": dbf, "Spider": SPDB}; CATS = {"BIRD": CB, "Spider": CS}; QS = {"BIRD": BQ_DEV, "Spider": SQ}
    TITLES = {"BIRD": "BIRD dev, with evidence (RASL's setting)", "Spider": "Spider dev"}; TAGS = {"BIRD": "bird_dev_ev", "Spider": "spider_dev"}
    DDL = {}
    def ddl(ds, db):
        if (ds, db) not in DDL:
            DDL[(ds, db)] = {}
            if db in DBFILE[ds]:
                con = sqlite3.connect(f"file:{DBFILE[ds][db]}?mode=ro", uri=True)
                for n_, s_ in con.execute("SELECT name, sql FROM sqlite_master WHERE type='table'"):
                    if s_: DDL[(ds, db)][n_.lower()] = s_.strip()
                con.close()
        return DDL[(ds, db)]
    say("EX 1/3: prompts from our rankings")
    JOBS = []
    for ds in ("BIRD", "Spider"):
        if ds.lower() not in EX_SETS: continue
        C_, Q_ = CATS[ds], QS[ds]; X_, y_, g_, c_ = M[TAGS[ds]]; sc_ = PRED[TITLES[ds]]
        for i in range(len(Q_) if not EX_N else min(EX_N, len(Q_))):
            ix = np.where(g_ == i)[0]; order = c_[ix[np.argsort(-sc_[ix])]]; db = C_["TABLES"][order[0]][0]
            for k in (5, 15):
                JOBS.append(dict(ds=ds, cond=f"ours, top {k}", i=i, db=db, ids=[int(j) for j in order[:k] if C_["TABLES"][j][0] == db]))
            JOBS.append(dict(ds=ds, cond="gold tables", i=i, db=Q_[i]["db"], ids=sorted(int(j) for j in Q_[i]["gold"])))

    def messages(job):
        C_, q = CATS[job["ds"]], QS[job["ds"]][job["i"]]; d = ddl(job["ds"], job["db"])
        schema = "\n\n".join(d.get(C_["TABLES"][j][1].lower(), C_["CARD"][j]) for j in job["ids"])
        ev = f"\nEvidence: {q['evidence']}" if q["evidence"] else ""
        return [{"role": "system", "content": "You are an expert data analyst who writes correct SQLite queries."},
                {"role": "user", "content": f"Database schema (SQLite):\n\n{schema}\n\nQuestion: {q['question']}{ev}\n\n"
                                            "Write one SQLite query that answers the question. Return only the query in a ```sql code block."}]

    RUNNER = r'''
import json, sys, os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
a = json.loads(open(sys.argv[1]).read())
tok = AutoTokenizer.from_pretrained(a["model"])
rows = [json.loads(l) for l in open(a["inp"])]
prompts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
limit = a["max_model_len"] - a["max_tokens"]
ok = [i for i, p in enumerate(prompts) if len(tok(p)["input_ids"]) <= limit]
print(f"[runner] {len(prompts)} prompts, {len(prompts) - len(ok)} too long (skipped) | loading {a['model']} with {a['llm_kwargs']}", flush=True)
llm = LLM(model=a["model"], max_model_len=a["max_model_len"], gpu_memory_utilization=a["gpu"], seed=0, **a["llm_kwargs"])
sp = SamplingParams(temperature=0.0, max_tokens=a["max_tokens"], seed=0)
text = [""] * len(prompts)
for c in range(0, len(ok), 256):
    idx = ok[c:c + 256]
    for i, o in zip(idx, llm.generate([prompts[i] for i in idx], sp, use_tqdm=False)): text[i] = o.outputs[0].text
    print(f"[runner] generated {min(c + 256, len(ok))}/{len(ok)}", flush=True)
with open(a["out"], "w") as f:
    for t in text: f.write(json.dumps({"text": t, "model": a["model"]}) + "\n")
'''
    Path("/content/vllm_run.py").write_text(RUNNER)
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    G31, G26 = "google/gemma-4-31B-it", "google/gemma-4-26B-A4B-it"
    ATTEMPTS = ([(G31, {}), (G31, {"quantization": "fp8"})] if gpu_gb > 70 else
                [(G31, {"quantization": "fp8", "limit_mm_per_prompt": {"image": 0}}), (G31, {"quantization": "fp8"}),
                 (G26, {"quantization": "fp8", "limit_mm_per_prompt": {"image": 0}}), (G26, {"quantization": "fp8"})])
    out_f = OUT / f"ex_generations_{'_'.join(EX_SETS)}_{EX_N}.jsonl"
    if not (out_f.exists() and sum(1 for _ in open(out_f)) == len(JOBS)):
        say(f"EX 2/3: Gemma writes {len(JOBS)} queries (GPU {gpu_gb:.0f} GB; first load downloads the weights)")
        inp = Path("/content/ex_in.jsonl"); inp.write_text("".join(json.dumps({"messages": messages(j)}) + "\n" for j in JOBS))
        for k, (model, kw) in enumerate(ATTEMPTS, 1):
            cfg = Path("/content/ex_cfg.json"); cfg.write_text(json.dumps(dict(model=model, inp=str(inp), out=str(out_f), max_model_len=8192,
                                                                                max_tokens=512, gpu=0.90, llm_kwargs=kw)))
            p_ = subprocess.Popen([sys.executable, "/content/vllm_run.py", str(cfg)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  env={**os.environ, "VLLM_LOGGING_LEVEL": "WARNING", "TOKENIZERS_PARALLELISM": "false"})
            lines = []
            for line in p_.stdout:
                lines.append(line.rstrip())
                if line.startswith("[runner]"): say("  " + line.rstrip())
            (OUT / f"ex_log_attempt{k}.txt").write_text("\n".join(lines))
            if p_.wait() == 0: break
            cause = [l for l in lines if re.search(r"(OutOfMemory|out of memory|CUDA error|ValueError|RuntimeError|KV cache|Free memory|not supported|Error:)", l, re.I)]
            say(f"  attempt {k}/{len(ATTEMPTS)} failed ({model} {kw}); likely cause:"); print("\n".join("      " + l[:300] for l in (cause[-6:] or lines[-12:])))
        else:
            raise SystemExit(f"Gemma could not be loaded on this GPU; logs in {OUT}")
    GEN = [json.loads(l) for l in open(out_f)]
    say("EX 3/3: executing")
    def extract(t):
        b = re.findall(r"```sql\s*(.*?)```", t, re.S)
        return (b[-1] if b else t.split("<channel|>")[-1]).strip()
    def run_sql(path, sql):
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True); con.text_factory = lambda b: b.decode(errors="ignore")
        try: return con.execute(sql).fetchall()
        finally: con.close()
    GOLDRES = {}
    def correct(job, sql):
        ds, q = job["ds"], QS[job["ds"]][job["i"]]
        if job["db"] not in DBFILE[ds] or q["db"] not in DBFILE[ds]: return 0.0
        try:
            if (ds, job["i"]) not in GOLDRES: GOLDRES[(ds, job["i"])] = set(func_timeout(30, run_sql, args=(DBFILE[ds][q["db"]], q["sql"])))
            return float(set(func_timeout(30, run_sql, args=(DBFILE[ds][job["db"]], sql))) == GOLDRES[(ds, job["i"])])
        except (FunctionTimedOut, Exception):
            return 0.0
    for j_, g_ in zip(JOBS, GEN): j_["sql"] = extract(g_["text"]); j_["ok"] = correct(j_, j_["sql"]); j_["empty"] = not g_["text"]
    with open(OUT / "ex_rows.jsonl", "w") as fh:
        for j_ in JOBS: fh.write(json.dumps({k: v for k, v in j_.items()}) + "\n")
    used = GEN[0]["model"] if GEN else "?"
    EXR = ["", f"APPENDIX: execution accuracy, SQL written by {used} (zero-shot, greedy, thinking off) from the retrieved tables",
           f"   {'dataset':8s} {'condition':16s} {'EX %':>6s} {'tables shown':>13s} {'right database':>15s} {'skipped (too long)':>19s}"]
    for ds in ("BIRD", "Spider"):
        for cond in ("ours, top 5", "ours, top 15", "gold tables"):
            R_ = [j_ for j_ in JOBS if j_["ds"] == ds and j_["cond"] == cond]
            if not R_: continue
            right = np.mean([j_["db"] == QS[ds][j_["i"]]["db"] for j_ in R_])
            EXR.append(f"   {ds:8s} {cond:16s} {100*np.mean([j_['ok'] for j_ in R_]):6.1f} {np.mean([len(j_['ids']) for j_ in R_]):13.1f} "
                       f"{100*right:14.1f}% {sum(j_['empty'] for j_ in R_):19d}")
    EXR += ["   published, RASL with Claude 3.5 Sonnet writing the SQL: BIRD N=15 53.5 (gold tables 54.0) | Spider N=15 64.5 (gold tables 69.0)",
            "   note: the gold-tables row is the generator's ceiling; the gap between it and our rows is what retrieval still costs."]
    TIMES["EX appendix"] = time.time() - t_ex
    (OUT / "EX_APPENDIX.txt").write_text("\n".join(EXR) + "\n")
    print("\n".join(EXR))

# ================================================================ REVIEW ANALYSES (from saved outputs; no new data, no new models beyond LightGBM refits)
if RUN_ANALYSIS:
    t_an = time.time()
    AN = [f"Review analyses | {time.strftime('%Y-%m-%d %H:%M')} | frozen configuration, sqlglot labels", ""]
    LGB = dict(objective="lambdarank", n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.7, reg_lambda=1.0, random_state=0, verbose=-1)
    Xtr, ytr, gtr, ctr = M["bird_train_ev"]; Xtr = with_emb(Xtr, SB_ev_tr, gtr, ctr, pca)
    Xdv, ydv, gdv, cdv = M["bird_dev_ev"]; Xdv = with_emb(Xdv, SB_ev_dev, gdv, cdv, pca)
    ngt = np.array([len(q["gold"]) for q in BQ_TR]); ngd = np.array([len(q["gold"]) for q in BQ_DEV])
    FAMS = {}
    for n in ALL: FAMS.setdefault(n.split(":")[0], []).append(n)
    RETR = [n for n in ALL if n.split(":")[0] in ("score", "agree", "db")]
    OTHER = [n for n in ALL if n not in RETR]
    SETS_A = {f"retrieval signals only: score, agree, db ({len(RETR)})": RETR, f"other families only, no score/agree/db ({len(OTHER)})": OTHER,
              f"all features, FERN-ev ({len(ALL)})": ALL}
    for fam, fs in FAMS.items(): SETS_A[f"all minus {fam} ({len(fs)})"] = [n for n in ALL if n not in fs]
    def fit(cols_, X, y, g, rows=None):
        ix = [col[n] for n in cols_]
        if rows is None: return LGBMRanker(**LGB).fit(X[:, ix], y, group=np.bincount(g))
        gg = g[rows]; return LGBMRanker(**LGB).fit(X[rows][:, ix], y[rows], group=np.bincount(gg)[np.unique(gg)])
    def per_q(score, y, g, nq, ng, k):
        """recall and complete at k per question (arrays of length nq)."""
        order = np.lexsort((-score, g)); ys = y[order]; gs = g[order]
        start = np.searchsorted(gs, np.arange(nq)); end = np.searchsorted(gs, np.arange(nq), side="right")
        hit = np.array([ys[a:min(a + k, e)].sum() for a, e in zip(start, end)])
        return hit / ng, (hit >= ng).astype(float)
    # ---- 1  ablations: 5-fold CV on BIRD train (folds by database) for the three main sets; BIRD dev for all sets
    say("analysis 1/6: frozen-configuration ablations (CV on BIRD train; ~30 minutes)")
    dbs_tr = sorted({q["db"] for q in BQ_TR}); random.Random(0).shuffle(dbs_tr); FO = {d: k % 5 for k, d in enumerate(dbs_tr)}
    qfold = np.array([FO[q["db"]] for q in BQ_TR]); rfold = qfold[gtr]
    CVRES, DEVRES = {}, {}
    for nm, fs in SETS_A.items():
        if nm.startswith(("retrieval", "other", "all features")):
            sc = np.zeros(len(ytr))
            for k in range(5):
                m_ = fit(fs, Xtr, ytr, gtr, rows=np.where(rfold != k)[0]); te = rfold == k
                sc[te] = m_.predict(Xtr[te][:, [col[n] for n in fs]])
            CVRES[nm] = {k: per_q(sc, ytr, gtr, len(BQ_TR), ngt, k) for k in (5, 15)}
        m_ = fit(fs, Xtr, ytr, gtr); sd = m_.predict(Xdv[:, [col[n] for n in fs]])
        DEVRES[nm] = {k: per_q(sd, ydv, gdv, len(BQ_DEV), ngd, k) for k in (5, 15)}
        say(f"  {nm}")
    def pb(a, b):
        d = a - b; bs = [d[rs.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        return f"{100*d.mean():+5.1f} [{100*np.percentile(bs, 2.5):+.1f}, {100*np.percentile(bs, 97.5):+.1f}]"
    full = [k for k in SETS_A if k.startswith("all features")][0]
    AN += ["== 1  Ablations in the frozen configuration (FERN-ev inputs, BIRD with evidence; lambdarank, same hyperparameters)",
           f"   {'feature set':48s} {'CV R@5':>7s} {'CV C@5':>7s} {'CV R@15':>8s} {'CV C@15':>8s} | {'dev R@5':>7s} {'dev C@5':>7s} {'dev R@15':>8s} {'dev C@15':>8s}"]
    for nm in SETS_A:
        cvs = " ".join(f"{100*CVRES[nm][k][j].mean():7.1f}" if nm in CVRES else f"{'':7s}" for k, j in ((5, 0), (5, 1), (15, 0), (15, 1)))
        dvs = " ".join(f"{100*DEVRES[nm][k][j].mean():7.1f}" for k, j in ((5, 0), (5, 1), (15, 0), (15, 1)))
        AN.append(f"   {nm:48s} {cvs} | {dvs}")
    AN += ["", "   paired differences, all features minus subset [95% bootstrap over questions]:"]
    for nm in SETS_A:
        if nm == full: continue
        src = CVRES if nm in CVRES else None
        cvpart = (f"CV C@5 {pb(CVRES[full][5][1], CVRES[nm][5][1])}, CV C@15 {pb(CVRES[full][15][1], CVRES[nm][15][1])} | " if src else "")
        AN.append(f"   {nm:48s} {cvpart}dev C@5 {pb(DEVRES[full][5][1], DEVRES[nm][5][1])}, dev R@15 {pb(DEVRES[full][15][0], DEVRES[nm][15][0])}")
    # ---- 2  without cell values: FERN-schema on BIRD dev and Spider, and an evidence variant without values/types/row counts
    say("analysis 2/6: variants without cell values")
    NOVAL_EV = [n for n in ALL if n.split(":")[0] not in ("val", "type") and n != "meta:log_rows"]
    m_nv = fit(NOVAL_EV, Xtr, ytr, gtr); s_nv = m_nv.predict(Xdv[:, [col[n] for n in NOVAL_EV]])
    AN += ["", "== 2  Without cell values (RASL does not index database values)", f"   {'setting':62s} {'R@5':>6s} {'R@15':>6s} {'C@5':>6s} {'C@15':>6s}"]
    def row(nm, score, y, g, Q_):
        ng = np.array([len(q["gold"]) for q in Q_]); r5, c5 = per_q(score, y, g, len(Q_), ng, 5); r15, c15 = per_q(score, y, g, len(Q_), ng, 15)
        AN.append(f"   {nm:62s} {100*r5.mean():6.1f} {100*r15.mean():6.1f} {100*c5.mean():6.1f} {100*c15.mean():6.1f}")
    row("BIRD dev, evidence, all features (FERN-ev)", PRED["BIRD dev, with evidence (RASL's setting)"], ydv, gdv, BQ_DEV)
    row(f"BIRD dev, evidence, no values/types/row counts ({len(NOVAL_EV)}, refit on BIRD train)", s_nv, ydv, gdv, BQ_DEV)
    Xq, yq, gq, cq = M["bird_dev_noev"]; Xq = with_emb(Xq, SB_no_dev, gq, cq, pca)
    row("BIRD dev, question only, FERN-q", PRED["BIRD dev, question only"], yq, gq, BQ_DEV)
    row("BIRD dev, question only, FERN-schema (already fitted)", FUS["F_schema"].predict(Xq[:, [col[n] for n in FSETS["F_schema"]]]), yq, gq, BQ_DEV)
    Xs, ys_, gs_, cs_ = M["spider_dev"]; Xs = with_emb(Xs, SS, gs_, cs_, pca)
    row("Spider dev, FERN-q", PRED["Spider dev"], ys_, gs_, SQ)
    row("Spider dev, FERN-schema (already fitted)", FUS["F_schema"].predict(Xs[:, [col[n] for n in FSETS["F_schema"]]]), ys_, gs_, SQ)
    # ---- 3  recall@k for k = 1..20
    say("analysis 3/6: recall@k curves")
    AN += ["", "== 3  Recall@k and complete@k, k = 1 to 20 (RASL reports R@5 and R@15 only; its N is a maximum budget)",
           "   k   " + "  ".join(f"{h:>14s}" for h in ("BIRD-ev R / C", "BIRD-q R / C", "Spider R / C", "pool BIRD-ev R"))]
    pool_b = Xdv[:, col["score:pool"]]
    for k in list(range(1, 11)) + [12, 15, 20]:
        a = per_q(PRED["BIRD dev, with evidence (RASL's setting)"], ydv, gdv, len(BQ_DEV), ngd, k)
        b = per_q(PRED["BIRD dev, question only"], yq, gq, len(BQ_DEV), ngd, k)
        c = per_q(PRED["Spider dev"], ys_, gs_, len(SQ), np.array([len(q["gold"]) for q in SQ]), k)
        p_ = per_q(pool_b, ydv, gdv, len(BQ_DEV), ngd, k)
        AN.append(f"   {k:<3d} " + "  ".join(f"{100*x[0].mean():6.1f} / {100*x[1].mean():5.1f}" for x in (a, b, c)) + f"  {100*p_[0].mean():12.1f}")
    # ---- 4  Fiben: is the first stage the bottleneck?
    say("analysis 4/6: Fiben candidate pool")
    AN += ["", "== 4  Fiben: recall of the first stage over the whole catalog (152 tables) vs FERN"]
    for nm, key in (("BM25", "bm25_full"), ("Qwen3", "dense_e1"), ("BGE", "dense_e2"), ("pool (BM25 + Qwen3 + BGE)", "pool")):
        o = np.argsort(-SF[key], 1)
        AN.append(f"   {nm:28s} " + "  ".join(f"R@{k} {100*np.mean([len(q['gold'] & set(o[i, :k])) / len(q['gold']) for i, q in enumerate(FQ)]):5.1f}" for k in (15, 50, 100)))
    Xf_, yf_, gf_, cf_ = M["fiben"]; Xf_ = with_emb(Xf_, SF, gf_, cf_, pca)
    nf = np.array([len(q["gold"]) for q in FQ]); sf_ = FUS["F_schema"].predict(Xf_[:, [col[n] for n in FSETS["F_schema"]]])
    AN.append("   FERN-schema                  " + "  ".join(f"R@{k} {100*per_q(sf_, yf_, gf_, len(FQ), nf, k)[0].mean():5.1f}" for k in (15, 50, 100)))
    n_extra = len(FQ) - 279; r15 = per_q(sf_, yf_, gf_, len(FQ), nf, 15)[0].mean(); r5 = per_q(sf_, yf_, gf_, len(FQ), nf, 5)[0].mean()
    AN.append(f"   bound for any 279-question subset: R@15 {100*(len(FQ)*r15 - n_extra)/279:.1f} to {100*len(FQ)*r15/279:.1f}, "
              f"R@5 {100*(len(FQ)*r5 - n_extra)/279:.1f} to {100*len(FQ)*r5/279:.1f} (RASL 69.2 / 69.1)")
    # ---- 5  database-clustered bootstrap for the headline numbers
    say("analysis 5/6: database-clustered bootstrap")
    AN += ["", "== 5  Headline recall with intervals by question and by database (resampling whole databases)"]
    for nm, score, y, g, Q_ in (("BIRD dev, evidence", PRED["BIRD dev, with evidence (RASL's setting)"], ydv, gdv, BQ_DEV),
                                ("Spider dev", PRED["Spider dev"], ys_, gs_, SQ)):
        ng = np.array([len(q["gold"]) for q in Q_]); dbq = np.array([q["db"] for q in Q_]); ud = sorted(set(dbq)); idx = {d: np.where(dbq == d)[0] for d in ud}
        for k in (5, 15):
            r = per_q(score, y, g, len(Q_), ng, k)[0]
            bq = [r[rs.integers(0, len(r), len(r))].mean() for _ in range(2000)]
            bd = [np.concatenate([idx[d] for d in rs.choice(ud, len(ud))]) for _ in range(2000)]; bd = [r[s_].mean() for s_ in bd]
            AN.append(f"   {nm:20s} R@{k:<2d} {100*r.mean():5.1f}  question CI [{100*np.percentile(bq, 2.5):.1f}, {100*np.percentile(bq, 97.5):.1f}]  "
                      f"database CI [{100*np.percentile(bd, 2.5):.1f}, {100*np.percentile(bd, 97.5):.1f}]  ({len(ud)} databases)")
    # ---- 6  execution accuracy on correctly routed questions (saved generations)
    say("analysis 6/6: execution accuracy by routing")
    exf = OUT / "ex_rows.jsonl"
    if exf.exists():
        ER = [json.loads(l) for l in open(exf)]
        AN += ["", "== 6  Execution accuracy (Gemma-4-26B-A4B) on all questions and on questions routed to the right database",
               f"   {'dataset':8s} {'condition':14s} {'EX all':>7s} {'EX right db':>12s} {'gold tables, same questions':>28s} {'n right db':>11s}"]
        for ds in ("BIRD", "Spider"):
            gold = {r["i"]: r["ok"] for r in ER if r["ds"] == ds and r["cond"] == "gold tables"}
            for cond in ("ours, top 5", "ours, top 15"):
                R_ = [r for r in ER if r["ds"] == ds and r["cond"] == cond]
                if not R_: continue
                QS_ = BQ_DEV if ds == "BIRD" else SQ
                right = [r for r in R_ if r["db"] == QS_[r["i"]]["db"]]
                AN.append(f"   {ds:8s} {cond:14s} {100*np.mean([r['ok'] for r in R_]):7.1f} {100*np.mean([r['ok'] for r in right]):12.1f} "
                          f"{100*np.mean([gold[r['i']] for r in right]):28.1f} {len(right):11d}")
    TIMES["analysis"] = time.time() - t_an
    (OUT / "ANALYSIS.txt").write_text("\n".join(AN) + "\n")
    print("\n" + "\n".join(AN))
