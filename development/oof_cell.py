# ============ ONE CELL: no-LLM fuser, full version. Out-of-fold over all BIRD train databases, ~140 features. A100 40GB is fine. ============
# Question: can a feature-engineered fuser (no LLM anywhere) match or beat RASL on BIRD dev table recall?
# RASL, same catalog (80 databases, 597 tables, 4,337 columns) and same 1,534 dev questions:
#   recall@15 92.6 retrieval only, 97.8 after Claude 3.5 Sonnet selects tables.
# Pipeline: hybrid search top 100 -> features (13 families) -> fuser (LightGBM lambdarank, and a stacked ensemble).
# Out of fold: the 69 train databases are split into 5 folds; for each fold a decider (OpenDecider-nano) is trained on the other 4
# and scores the held-out fold, so the fuser never learns from a decider that saw the same questions. Dev databases are never used.
# Leakage guard: features are computed from (question, evidence, catalog, cell values) only. The gold SQL and gold database are
# never passed to the feature code. The report flags any single feature that predicts the label suspiciously well,
# and FEATURE_DUMP.csv (open it in Sheets) shows every feature value for 30 dev questions.
# About 1.5 to 2 hours the first time (6 deciders and the value index are the slow parts); every stage is cached in OUT_DIR.
OUT_DIR = "/content/drive/MyDrive/nift_rasl/oof"
N_FOLDS = 5
CAP_TRAIN_Q = 2500        # questions used to train each fold's decider (sampled from the other folds)
CAP_FIT_Q = 600           # questions per fold used to fit the fuser (so up to 3,000 in total)
N_DEV_Q = 0               # 0 = all 1,534
K_CAND = 100
USE_EVIDENCE = True       # BIRD's evidence hint is part of the question (RASL's examples appear to include it). False = question only.
USE_VALUES = True         # cell values from the sqlite databases (downloads the BIRD train databases once, several GB)
LOSS = "mix"              # decider loss: "mix" = BCE + listwise, "bce" = BCE only
EMB1, EMB2 = "Qwen/Qwen3-Embedding-0.6B", "BAAI/bge-large-en-v1.5"
DECIDER = "manjunathshiva/opendecider-nano"
RUN_MAIN = False          # the main comparison (A, C, D, E, drop-one-family), already run; True to rerun it
RUN_EXTRAS = False        # decider ablation, frozen-scorers-only fuser, sqlglot gold labels, set sizes, timing per query (done)
RUN_BEAVER = False        # zero-shot transfer to the current BEAVER release (done)
RUN_SELECT = True         # scorer selection by cross-validation on the training questions (rules fixed in advance, see below)
BEAVER_DIR = "/content/drive/MyDrive/beaver_v3"                     # parquet files from the BEAVER download cell
KEYS_FILE = "/content/drive/MyDrive/nift_v2/schema_keys.json"       # declared foreign keys (optional)
# =====================================================================================================================
import os, subprocess, sys, time
T0 = time.time()
def say(msg): print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)
say("step 1/9: installing (1 to 2 minutes)")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rank_bm25", "sentence-transformers", "lightgbm", "remotezip", "rapidfuzz", "sqlglot", "pandas", "pyarrow"], check=True)
from google.colab import drive
drive.mount("/content/drive")
import json, re, zipfile, random, glob, pickle, sqlite3
from pathlib import Path
import numpy as np
import torch, gc
OUT = Path(OUT_DIR); OUT.mkdir(parents=True, exist_ok=True)
TAG = f"f{N_FOLDS}_t{CAP_TRAIN_Q}_q{CAP_FIT_Q}_{LOSS}_{'ev' if USE_EVIDENCE else 'noev'}"
TIMES = {}

# ---------------------------------------------------------------- 2  BIRD data
say("step 2/9: BIRD data")
DEV = Path("/content/bird_dev"); TRN = Path("/content/bird_train"); DEV.mkdir(exist_ok=True); TRN.mkdir(exist_ok=True)
BIRD = "https://bird-bench.oss-cn-beijing.aliyuncs.com/"
if not glob.glob(str(DEV / "**/dev_databases"), recursive=True) or not (DEV / "dev.json").exists():
    subprocess.run(["wget", "-q", "--show-progress", "--progress=bar:force:noscroll", "-O", "/content/dev.zip", BIRD + "dev.zip"], check=True)
    with zipfile.ZipFile("/content/dev.zip") as z: z.extractall("/content/bird_raw")
    root = Path(glob.glob("/content/bird_raw/**/dev.json", recursive=True)[0]).parent
    for f in ("dev.json", "dev_tables.json"): (DEV / f).write_bytes((root / f).read_bytes())
    inner = glob.glob(str(root / "dev_databases*.zip"))
    if inner:
        with zipfile.ZipFile(inner[0]) as z: z.extractall(DEV)
    else:
        subprocess.run(["cp", "-r", str(root / "dev_databases"), str(DEV)], check=True)
need_trn_db = USE_VALUES and not (OUT / "values_index.pkl").exists() and not glob.glob(str(TRN / "**/train_databases"), recursive=True)
if need_trn_db or not (TRN / "train.json").exists():
    if need_trn_db:
        say("  downloading BIRD train with its databases (several GB, once)")
        subprocess.run(["wget", "-q", "--show-progress", "--progress=bar:force:noscroll", "-O", "/content/train.zip", BIRD + "train.zip"], check=True)
        with zipfile.ZipFile("/content/train.zip") as z:
            real = [n for n in z.namelist() if "__MACOSX" not in n and not Path(n).name.startswith("._")]
            for n in real:
                if Path(n).name in ("train.json", "train_tables.json"): (TRN / Path(n).name).write_bytes(z.read(n))
            inner = [n for n in real if Path(n).name == "train_databases.zip"]
            if inner: z.extract(inner[0], "/content/trn_raw")
            else: z.extractall("/content/trn_raw")
        if inner:
            say("  unzipping train databases")
            with zipfile.ZipFile(glob.glob("/content/trn_raw/**/train_databases.zip", recursive=True)[0]) as z: z.extractall(TRN)
        else:
            subprocess.run(["cp", "-r", glob.glob("/content/trn_raw/**/train_databases", recursive=True)[0], str(TRN)], check=True)
        os.remove("/content/train.zip")
    else:
        from remotezip import RemoteZip
        with RemoteZip(BIRD + "train.zip") as z:
            for want in ("train.json", "train_tables.json"):
                (TRN / want).write_bytes(z.read([n for n in z.namelist() if n.endswith("/" + want) or n == want][0]))
DEV_Q_ALL = json.loads((DEV / "dev.json").read_text())
TRN_Q_ALL = json.loads((TRN / "train.json").read_text())
SCHEMAS = json.loads((TRN / "train_tables.json").read_text()) + json.loads((DEV / "dev_tables.json").read_text())
DBFILE = {}
for base in [glob.glob(str(DEV / "**/dev_databases"), recursive=True), glob.glob(str(TRN / "**/train_databases"), recursive=True)]:
    for b in base:
        for f in glob.glob(b + "/*/*.sqlite"): DBFILE[Path(f).parent.name] = f

# ---------------------------------------------------------------- 3  catalog: tables, columns, types, keys, foreign-key graph
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

NUMT = ("int", "real", "num", "float", "double", "decimal")
TABLES, CARD, NAMECARD, COLCARD, COLS, NCOL, CTYPE, NPK, NFKC = [], [], [], [], [], [], [], [], []
ADJ, COLTXT = {}, []
for s in SCHEMAS:
    db, tn = s["db_id"], s["table_names_original"]
    cid = {i: (t, c) for i, (t, c) in enumerate(s["column_names_original"])}
    pk = set()
    for p in s["primary_keys"]: pk |= set(p) if isinstance(p, list) else {p}
    fkc = {a for a, b in s["foreign_keys"]} | {b for a, b in s["foreign_keys"]}
    base = len(TABLES)
    for ti, t in enumerate(tn):
        mine = [(i, c, nat, ty) for i, ((tt, c), (_, nat), ty) in enumerate(zip(s["column_names_original"], s["column_names"], s["column_types"])) if tt == ti]
        desc = [c if nat.lower().replace(" ", "") == c.lower().replace("_", "") else f"{c} ({nat})" for _, c, nat, _ in mine]
        TABLES.append((db, t)); CARD.append(f"{t} (database {db.replace('_', ' ')}): " + ", ".join(desc))
        NAMECARD.append(f"{camel(t).replace('_', ' ')} (database {db.replace('_', ' ')})"); COLCARD.append(", ".join(desc))
        COLS.append([(c, nat) for _, c, nat, _ in mine]); NCOL.append(len(mine)); CTYPE.append([ty.lower() for *_, ty in mine])
        NPK.append(sum(i in pk for i, *_ in mine)); NFKC.append(sum(i in fkc for i, *_ in mine))
        for _, c, nat, _ in mine: COLTXT.append((len(TABLES) - 1, f"{t} . {c} ({nat})"))
        ADJ[len(TABLES) - 1] = set()
    for a, b in s["foreign_keys"]:
        ta, tb = base + cid[a][0], base + cid[b][0]
        if ta != tb: ADJ[ta].add(tb); ADJ[tb].add(ta)
NT = len(TABLES)
TBL_OF = {}
for i, (d, t) in enumerate(TABLES): TBL_OF.setdefault(d, []).append(i)
TTOK = [set(toks(t)) for _, t in TABLES]
CTOK = [set(toks(t) + [w for c, nat in COLS[i] for w in toks(c) + toks(nat)]) for i, (_, t) in enumerate(TABLES)]
CNAMES = [{c.lower() for c, _ in COLS[i]} - {"id", "name", "type", "date", "year", "description", "code"} for i in range(NT)]
DF = {}
for ts in CTOK:
    for w in ts: DF[w] = DF.get(w, 0) + 1
IDF = {w: np.log(NT / (1 + d)) for w, d in DF.items()}
VOCAB = sorted(DF); VID = {w: i for i, w in enumerate(VOCAB)}
# PageRank on each database's foreign-key graph
PR = np.zeros(NT)
for d, ids in TBL_OF.items():
    n = len(ids); loc = {j: k for k, j in enumerate(ids)}; r = np.ones(n) / n
    for _ in range(30):
        nr = np.full(n, 0.15 / n)
        for j in ids:
            nb = ADJ[j]
            for b in nb: nr[loc[b]] += 0.85 * r[loc[j]] / len(nb)
            if not nb: nr += 0.85 * r[loc[j]] / n
        r = nr
    for j in ids: PR[j] = r[loc[j]] * n
print(f"catalog: {len(TBL_OF)} databases, {NT} tables, {len(COLTXT)} columns (RASL: 80, 597, 4,337)")


def gold_ids(q):                                                          # labels only; never passed to the feature code
    sql = q["SQL"].lower()
    return [i for i in TBL_OF[q["db_id"]] if re.search(r'(?<![\w])[`"\[]?' + re.escape(TABLES[i][1].lower()) + r'[`"\]]?(?![\w])', sql)]


pub = lambda q: {"question": q["question"], "evidence": (q.get("evidence", "") or "") if USE_EVIDENCE else ""}
qtext = lambda p: (p["question"] + " " + p["evidence"]).strip()

# ---------------------------------------------------------------- 4  cell values: distinct text values, numeric ranges, row counts
VAL_F = OUT / "values_index.pkl"
if USE_VALUES and VAL_F.exists():
    VALS, NUMR, NROWS = pickle.loads(VAL_F.read_bytes())
elif USE_VALUES:
    say("step 3/9: building the value index from the sqlite databases (once, 10 to 20 minutes)")
    t0 = time.time(); VALS, NUMR, NROWS = {}, {}, {}
    for d, ids in TBL_OF.items():
        if d not in DBFILE: continue
        con = sqlite3.connect(f"file:{DBFILE[d]}?mode=ro", uri=True); con.text_factory = lambda b: b.decode(errors="ignore")
        deadline = [0]
        con.set_progress_handler(lambda: 1 if time.time() > deadline[0] else 0, 20000)
        for j in ids:
            t = TABLES[j][1]
            try:
                deadline[0] = time.time() + 5; NROWS[j] = con.execute(f'SELECT max(rowid) FROM "{t}"').fetchone()[0] or 0
            except Exception:
                NROWS[j] = 0
            for (c, _), ty in zip(COLS[j], CTYPE[j]):
                try:
                    deadline[0] = time.time() + 5
                    if any(k in ty for k in NUMT):
                        lo, hi = con.execute(f'SELECT min("{c}"), max("{c}") FROM "{t}"').fetchone()
                        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)): NUMR[(j, c)] = (lo, hi)
                    else:
                        vs = con.execute(f'SELECT DISTINCT "{c}" FROM "{t}" WHERE "{c}" IS NOT NULL LIMIT 20000').fetchall()
                        VALS[(j, c)] = {str(v[0]).strip().lower() for v in vs if v[0] is not None and 1 < len(str(v[0]).strip()) <= 60}
                except Exception:
                    pass
        con.close(); say(f"  values: {d} done")
    VAL_F.write_bytes(pickle.dumps((VALS, NUMR, NROWS)))
    TIMES["value index"] = time.time() - t0
else:
    VALS, NUMR, NROWS = {}, {}, {}
VCOLS = {}
for (j, c) in VALS: VCOLS.setdefault(j, []).append(c)
NCOLS_R = {}
for (j, c) in NUMR: NCOLS_R.setdefault(j, []).append(c)
CAPS_STOP = {"what", "how", "list", "which", "please", "among", "give", "name", "calculate", "state", "indicate", "provide", "find", "identify",
             "in", "for", "the", "who", "when", "where", "is", "are", "of", "on", "a", "an", "and", "or", "by", "with", "from", "to", "at", "as", "if"}


def literals(p):
    text = p["question"] + " " + p["evidence"]
    lit = {a or b for a, b in re.findall(r"'([^']{2,60})'|\"([^\"]{2,60})\"", text)}
    for span in re.findall(r"\b([A-Z][\w\-\.']*(?:\s+[A-Z][\w\-\.']*)*)", p["question"]):
        ws = [w for w in span.split() if w.lower() not in CAPS_STOP]
        if ws: lit.add(" ".join(ws))
    nums = [float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])", text)]
    return {l.strip().lower() for l in lit if len(l.strip()) > 1}, nums


# ---------------------------------------------------------------- 5  scorers over the whole catalog (7 views), for every question
say("step 4/9: search scorers (2 embedders, BM25, 3 text views)")
t0 = time.time()
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
words = lambda s: re.findall(r"[a-z0-9]+", camel(s).lower().replace("_", " "))
rng = random.Random(0)
trn_dbs = sorted({q["db_id"] for q in TRN_Q_ALL}); rng.shuffle(trn_dbs)
FOLD = {d: k % N_FOLDS for k, d in enumerate(trn_dbs)}
TRQ = [q for q in TRN_Q_ALL if gold_ids(q)]
DEVQ = random.Random(3).sample(DEV_Q_ALL, N_DEV_Q) if N_DEV_Q else DEV_Q_ALL
ALLQ = TRQ + DEVQ; NTR = len(TRQ)
PUB = [pub(q) for q in ALLQ]
SC_F = OUT / f"scorers_{'ev' if USE_EVIDENCE else 'noev'}_{len(DEVQ)}.npz"
if SC_F.exists():
    z = np.load(SC_F); SC = {k: z[k] for k in z.files}
else:
    SC = {}
    for nm, docs in (("bm25_full", CARD), ("bm25_name", NAMECARD)):
        b = BM25Okapi([words(c) for c in docs]); SC[nm] = np.stack([b.get_scores(words(qtext(p))) for p in PUB]).astype(np.float32)
    for tag, mname in (("e1", EMB1), ("e2", EMB2)):
        emb = SentenceTransformer(mname, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
        kw = {"prompt_name": "query"} if tag == "e1" else {"prompt": "Represent this sentence for searching relevant passages: "}
        QE = emb.encode([qtext(p) for p in PUB], normalize_embeddings=True, batch_size=64, **kw)
        SC[f"dense_{tag}_full"] = (QE @ emb.encode(CARD, normalize_embeddings=True, batch_size=32).T).astype(np.float32)
        if tag == "e1":
            SC["dense_e1_name"] = (QE @ emb.encode(NAMECARD, normalize_embeddings=True, batch_size=64).T).astype(np.float32)
            CE = emb.encode([c[1] for c in COLTXT], normalize_embeddings=True, batch_size=64)
            COL_T = np.array([c[0] for c in COLTXT]); QC = QE @ CE.T
            SC["dense_e1_bestcol"] = np.stack([QC[:, COL_T == j].max(1) if (COL_T == j).any() else np.zeros(len(QC)) for j in range(NT)], 1).astype(np.float32)
            SC["qe"] = QE.astype(np.float16); SC["te"] = emb.encode(CARD, normalize_embeddings=True, batch_size=32).astype(np.float16)
            del QC, CE
        del emb; gc.collect(); torch.cuda.empty_cache()
    np.savez(SC_F, **SC)
rankm = lambda X: np.argsort(np.argsort(-X, 1), 1)
HYB = 1 / (60 + rankm(SC["dense_e1_full"])) + 1 / (60 + rankm(SC["bm25_full"]))
CAND = np.argsort(-HYB, 1)[:, :K_CAND]
TIMES["scorers"] = time.time() - t0

# ---------------------------------------------------------------- 6  deciders, out of fold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
CARDT = CARD


def train_decider(qidx, path):
    if (path / "config.json").exists(): return
    t0 = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(DECIDER, num_labels=1, trust_remote_code=True, ignore_mismatched_sizes=True)
    tok = AutoTokenizer.from_pretrained(DECIDER, trust_remote_code=True)
    r = random.Random(4); groups = []
    for i in qidx:
        g = set(gold_ids(ALLQ[i]))
        hard = [j for j in np.argsort(-HYB[i])[:40] if j not in g][:8]
        easy = r.sample([j for j in range(NT) if j not in g], 4)
        groups.append([(qtext(PUB[i]), CARDT[j], 1.0) for j in g] + [(qtext(PUB[i]), CARDT[j], 0.0) for j in hard + easy])
    r.shuffle(groups)
    model.cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    QPB = 2; steps = (len(groups) + QPB - 1) // QPB
    sch = get_linear_schedule_with_warmup(opt, steps // 10, steps); bce = torch.nn.BCEWithLogitsLoss()

    def listwise(lg, y):
        pos, neg = lg[y > 0.5], lg[y < 0.5]
        if not len(pos) or not len(neg): return lg.sum() * 0
        return (torch.logsumexp(torch.cat([pos[:, None], neg[None, :].expand(len(pos), -1)], 1), 1) - pos).mean()

    for s in range(steps):
        gb = groups[s * QPB:(s + 1) * QPB]; b = [p for grp in gb for p in grp]
        x = tok([p[0] for p in b], [p[1] for p in b], truncation=True, max_length=320, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(**x).logits[:, 0].float()
        y = torch.tensor([p[2] for p in b], device="cuda"); loss = bce(lg, y)
        if LOSS == "mix":
            cuts = np.cumsum([0] + [len(grp) for grp in gb])
            loss = loss + torch.stack([listwise(lg[a:c], y[a:c]) for a, c in zip(cuts[:-1], cuts[1:])]).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad()
        if s % 400 == 0 or s == steps - 1: say(f"    step {s + 1}/{steps} loss {loss.item():.3f}")
    model.save_pretrained(path); tok.save_pretrained(path)
    del model; gc.collect(); torch.cuda.empty_cache()
    TIMES["deciders (training)"] = TIMES.get("deciders (training)", 0) + time.time() - t0


@torch.no_grad()
def score(path, qidx):
    t0 = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(str(path), num_labels=1, trust_remote_code=True).cuda().eval()
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    out = {}
    for a in range(0, len(qidx), 3):                                       # 3 questions (300 pairs) per forward pass
        chunk = qidx[a:a + 3]
        pa = [(qtext(PUB[i]), CARDT[j]) for i in chunk for j in CAND[i]]
        x = tok([p[0] for p in pa], [p[1] for p in pa], truncation=True, max_length=320, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(**x).logits[:, 0].float().cpu().numpy()
        for k, i in enumerate(chunk): out[i] = lg[k * K_CAND:(k + 1) * K_CAND]
        if a % 900 == 0: say(f"    scored {a}/{len(qidx)}")
    del model; gc.collect(); torch.cuda.empty_cache()
    TIMES["deciders (scoring)"] = TIMES.get("deciders (scoring)", 0) + time.time() - t0
    return out


DEC = np.zeros((len(ALLQ), K_CAND), dtype=np.float32)
FITIDX = []
for f in range(N_FOLDS + 1):                                               # folds 0..4, then the dev decider (trained on all folds)
    is_dev = f == N_FOLDS
    tr = [i for i in range(NTR) if is_dev or FOLD[ALLQ[i]["db_id"]] != f]
    tr = random.Random(10 + f).sample(tr, min(CAP_TRAIN_Q, len(tr)))
    tgt = list(range(NTR, len(ALLQ))) if is_dev else random.Random(20 + f).sample(
        [i for i in range(NTR) if FOLD[ALLQ[i]["db_id"]] == f], min(CAP_FIT_Q, sum(FOLD[ALLQ[i]["db_id"]] == f for i in range(NTR))))
    cache = OUT / f"dec_scores_{TAG}_{'dev' if is_dev else f'fold{f}'}_{len(tgt)}.npz"
    if cache.exists():
        z = np.load(cache); DEC[z["idx"]] = z["D"]
    else:
        path = OUT / f"decider_{TAG}_{'dev' if is_dev else f'fold{f}'}"
        say(f"step 5/9: decider {'for dev (all train folds)' if is_dev else f'fold {f + 1}/{N_FOLDS}'}: train on {len(tr)}, score {len(tgt)}")
        train_decider(tr, path)
        sc = score(path, tgt); DEC[tgt] = np.stack([sc[i] for i in tgt])
        np.savez(cache, idx=np.array(tgt), D=DEC[tgt])
    if not is_dev: FITIDX += tgt
DEVIDX = list(range(NTR, len(ALLQ)))

# ---------------------------------------------------------------- 7  features (13 families). Inputs: public question view + catalog only.
say("step 6/9: features")
t0 = time.time()
from rapidfuzz.process import cdist
from rapidfuzz.distance import Indel
SCORERS = {"hyb": None, "dec": None, "bm25_full": "bm25_full", "bm25_name": "bm25_name", "dense_e1": "dense_e1_full",
           "dense_e1_name": "dense_e1_name", "dense_e2": "dense_e2_full", "bestcol": "dense_e1_bestcol"}
AGG = re.compile(r"\b(how many|number of|count|average|mean|total|sum|highest|lowest|most|least|maximum|minimum|top|largest|smallest|"
                 r"oldest|youngest|best|worst)\b")
CMP = re.compile(r"(more than|less than|greater|fewer|over|under|at least|at most|above|below|>|<|between)")
PCT = re.compile(r"\b(percent|percentage|ratio|rate|proportion|times)\b")
TIMEW = re.compile(r"\b(year|date|when|month|day|born|since|before|after|age|(19|20)\d\d)\b")
DESC = {}                                                                  # feature name -> what it measures


def bfs(src, dst):
    if src == dst: return 0
    seen, fr = {src}, [src]
    for d in range(1, 6):
        nx = []
        for a in fr:
            for b in ADJ[a]:
                if b == dst: return d
                if b not in seen: seen.add(b); nx.append(b)
        fr = nx
    return 9


def feats(i):
    """All features for question i's 100 candidates. Uses PUB[i] (question + evidence), the catalog, scores and cell values."""
    p, cand = PUB[i], CAND[i]
    S = {"hyb": HYB[i, cand], "dec": DEC[i]}
    for k, v in SCORERS.items():
        if v: S[k] = SC[v][i, cand]
    R = {k: np.argsort(np.argsort(-v)) for k, v in S.items()}
    Z = {k: (v - v.mean()) / (v.std() + 1e-6) for k, v in S.items()}
    TOP = {k: v.max() for k, v in S.items()}
    PCTL = {k: R[k] / len(cand) for k in S}
    dbs = [TABLES[j][0] for j in cand]
    db_stats = {}
    for k in ("hyb", "dec", "dense_e1"):
        sm = np.exp(Z[k] - Z[k].max()); sm /= sm.sum()
        mx, mean, mass, n = {}, {}, {}, {}
        for c, d in enumerate(dbs):
            mx[d] = max(mx.get(d, -1e9), S[k][c]); mean[d] = mean.get(d, 0) + S[k][c]; mass[d] = mass.get(d, 0) + sm[c]; n[d] = n.get(d, 0) + 1
        rk = {d: r for r, d in enumerate(sorted(mx, key=lambda d: -mx[d]))}
        order = {}
        for c in np.argsort(-S[k]): order.setdefault(dbs[c], []).append(c)
        rin = {c: r for d in order for r, c in enumerate(order[d])}
        db_stats[k] = (mx, {d: mean[d] / n[d] for d in n}, mass, rk, rin, n)
    top1 = {k: cand[np.argmax(S[k])] for k in ("hyb", "dec")}
    top5 = {k: set(cand[np.argsort(-S[k])[:5]]) for k in ("hyb", "dec")}
    top3h = list(cand[np.argsort(-S["hyb"])[:3]])
    ql, evl = p["question"].lower(), p["evidence"].lower()
    qt = set(toks(p["question"])); qtall = qt | set(toks(p["evidence"]))
    qw = [w for w in qtall if len(w) >= 4]
    best_tok = cdist(qw, VOCAB, scorer=Indel.normalized_similarity, workers=-1).max(0) if qw else np.zeros(len(VOCAB))
    lits, nums = literals(p)
    flags = dict(q_agg=float(bool(AGG.search(ql))), q_cmp=float(bool(CMP.search(ql + " " + evl))), q_pct=float(bool(PCT.search(ql + " " + evl))),
                 q_time=float(bool(TIMEW.search(ql))))
    hs = np.exp(Z["hyb"] - Z["hyb"].max()); hs /= hs.sum()
    qctx = {"q:n_words": len(ql.split()), "q:n_content_tokens": len(qtall), "q:n_literals": len(lits), "q:n_numbers": len(nums),
            "q:has_evidence": float(bool(p["evidence"])), "q:top_entropy": float(-(hs * np.log(hs + 1e-12)).sum()),
            "q:top_gap": float(np.sort(S["hyb"])[-1] - np.sort(S["hyb"])[-2]), **{f"q:{k}": v for k, v in flags.items()}}
    # which literals match in which candidate (for per-database uniqueness)
    vmatch = {}
    for c, j in enumerate(cand):
        m = {l for col in VCOLS.get(j, []) for l in lits if l in VALS[(j, col)]}
        vmatch[c] = m
    lit_db = {}
    for c, d in enumerate(dbs):
        for l in vmatch[c]: lit_db.setdefault((d, l), set()).add(c)
    cover1 = CTOK[top1["hyb"]]; cover3 = set().union(*[CTOK[j] for j in top3h])
    rows = []
    for c, j in enumerate(cand):
        d = dbs[c]; f = {}
        for k in S:                                                       # family score: every scorer, 5 transforms
            f[f"score:{k}"] = S[k][c]; f[f"score:{k}_rank"] = R[k][c]; f[f"score:{k}_z"] = Z[k][c]
            f[f"score:{k}_gap_best"] = S[k][c] - TOP[k]; f[f"score:{k}_pctl"] = PCTL[k][c]
        rk = np.array([R[k][c] for k in S])                               # family agree: do the scorers agree?
        f.update({"agree:rank_min": rk.min(), "agree:rank_max": rk.max(), "agree:rank_mean": rk.mean(), "agree:rank_std": rk.std(),
                  "agree:n_top5": float((rk < 5).sum()), "agree:n_top15": float((rk < 15).sum()),
                  "agree:dec_vs_hyb": R["dec"][c] - R["hyb"][c], "agree:dense_vs_bm25": R["dense_e1"][c] - R["bm25_full"][c],
                  "agree:e1_vs_e2": R["dense_e1"][c] - R["dense_e2"][c], "agree:name_vs_full": R["dense_e1_name"][c] - R["dense_e1"][c]})
        for k in ("hyb", "dec", "dense_e1"):                              # family db: the table's database as a whole
            mx, mean, mass, rkd, rin, n = db_stats[k]
            f.update({f"db:{k}_max": mx[d], f"db:{k}_mean": mean[d], f"db:{k}_mass": mass[d], f"db:{k}_rank": rkd[d],
                      f"db:{k}_rank_in_db": rin[c], f"db:{k}_minus_dbmax": S[k][c] - mx[d]})
        f.update({"db:n_candidates": db_stats["hyb"][5][d], "db:n_tables": len(TBL_OF[d])})
        ct = CTOK[j]; ids = [VID[w] for w in ct]                          # family lex
        f.update({"lex:tname_overlap": len(qtall & TTOK[j]) / max(1, len(TTOK[j])), "lex:col_overlap": len(qtall & ct) / max(1, len(ct)),
                  "lex:idf_overlap": sum(IDF[w] for w in qtall & ct), "lex:n_overlap": len(qtall & ct),
                  "lex:fuzzy_max": max((best_tok[x] for x in ids), default=0), "lex:fuzzy_n_over_0.8": float(sum(best_tok[x] > 0.8 for x in ids)),
                  "lex:col_fullname_in_q": float(sum(1 for cc, nat in COLS[j] if len(nat) > 3 and nat.lower() in ql)),
                  "lex:tname_in_q": float(TABLES[j][1].lower().replace("_", " ") in ql)})
        f.update({"cov:q_tokens_covered": len(qtall & ct) / max(1, len(qtall)),                       # family cov: covering the question
                  "cov:gain_over_top1": len((qtall & ct) - cover1) / max(1, len(qtall)),
                  "cov:gain_over_top3": len((qtall & ct) - cover3) / max(1, len(qtall))})
        if USE_EVIDENCE:                                                  # family ev
            f.update({"ev:table_named": float(bool(re.search(r"(?<![\w])" + re.escape(TABLES[j][1].lower()) + r"(?![\w])", evl))),
                      "ev:n_cols_named": float(sum(1 for cc, _ in COLS[j] if re.search(r"(?<![\w])" + re.escape(cc.lower()) + r"(?![\w])", evl))),
                      "ev:n_natnames_named": float(sum(1 for _, nat in COLS[j] if len(nat) > 3 and nat.lower() in evl))})
        uniq = sum(1 for l in vmatch[c] if lit_db.get((d, l)) == {c})     # family val: cell values
        inrange = sum(1 for col in NCOLS_R.get(j, []) for x in nums if NUMR[(j, col)][0] <= x <= NUMR[(j, col)][1])
        f.update({"val:n_literals_matched": len(vmatch[c]), "val:frac_literals_matched": len(vmatch[c]) / max(1, len(lits)),
                  "val:n_cols_with_match": float(sum(1 for col in VCOLS.get(j, []) if any(l in VALS[(j, col)] for l in lits))),
                  "val:unique_in_db": float(uniq), "val:numbers_in_range": float(inrange),
                  "val:frac_numbers_in_range": inrange / max(1, len(nums) * max(1, len(NCOLS_R.get(j, []))))})
        ty = CTYPE[j]; nn = sum(any(k in t for k in NUMT) for t in ty)    # family type: what the question asks vs column types
        nd = sum(("date" in t or "time" in t) for t in ty) + sum(any(w in cc.lower() for w in ("date", "year", "time")) for cc, _ in COLS[j])
        f.update({"type:n_numeric": nn, "type:frac_numeric": nn / max(1, NCOL[j]), "type:n_datelike": nd,
                  "type:cmp_x_numeric": flags["q_cmp"] * nn, "type:agg_x_numeric": flags["q_agg"] * nn, "type:time_x_date": flags["q_time"] * nd,
                  "type:pct_x_numeric": flags["q_pct"] * nn})
        f.update({"meta:n_cols": NCOL[j], "meta:n_pk": NPK[j], "meta:n_fk_cols": NFKC[j], "meta:log_rows": np.log1p(max(0, NROWS.get(j, 0) or 0)),
                  "meta:junction": float(NFKC[j] >= 2 and NCOL[j] <= 4), "meta:name_len": len(TABLES[j][1])})
        f.update({"graph:degree": len(ADJ[j]), "graph:pagerank": PR[j],                                # family graph: joins
                  "graph:dist_top1_hyb": bfs(top1["hyb"], j), "graph:dist_top1_dec": bfs(top1["dec"], j),
                  "graph:n_nbr_top5_hyb": len(ADJ[j] & top5["hyb"]), "graph:n_nbr_top5_dec": len(ADJ[j] & top5["dec"]),
                  "graph:bridge": float(len(ADJ[j] & (top5["hyb"] | top5["dec"])) >= 2),
                  "graph:shared_cols_top1": len(CNAMES[j] & CNAMES[top1["hyb"]]) if j != top1["hyb"] else 0,
                  "graph:shared_cols_top5": float(sum(len(CNAMES[j] & CNAMES[t]) > 0 for t in top5["hyb"] if t != j))})
        f.update(qctx)
        rows.append(f)
    return rows


def build(idx):
    X, y, g, meta = [], [], [], []
    names = None
    for n_, i in enumerate(idx):
        rows = feats(i); gold = set(gold_ids(ALLQ[i]))                    # the label is attached after the features are computed
        if names is None: names = list(rows[0])
        X += [[r[k] for k in names] for r in rows]; y += [int(j in gold) for j in CAND[i]]; g += [n_] * K_CAND; meta += [(i, j) for j in CAND[i]]
        if n_ % 1000 == 0: say(f"    features {n_}/{len(idx)}")
    X = np.array(X, dtype=np.float32)
    QT = SC["qe"][[m[0] for m in meta]].astype(np.float32) * SC["te"][[m[1] for m in meta]].astype(np.float32)
    return X, np.array(y), np.array(g), meta, names, QT


FEAT_F = OUT / f"features_{TAG}_{len(FITIDX)}_{len(DEVIDX)}.pkl"
if FEAT_F.exists():
    say("  features cached, loading")
    Xf, yf, gf, mf, Xd, yd, gd, md, NAMES, NAN_FIX = pickle.loads(FEAT_F.read_bytes())
else:
    Xf, yf, gf, mf, NAMES, QTf = build(FITIDX)
    Xd, yd, gd, md, _, QTd = build(DEVIDX)
    from sklearn.decomposition import PCA
    pca = PCA(16, random_state=0).fit(QTf[::7])
    Xf = np.hstack([Xf, pca.transform(QTf)]); Xd = np.hstack([Xd, pca.transform(QTd)]); del QTf, QTd
    NAMES = NAMES + [f"emb:pc{k}" for k in range(16)]
    bad = ~np.isfinite(np.vstack([Xf, Xd]))                               # any non-finite value is set to 0 and reported
    NAN_FIX = {NAMES[k]: int(bad[:, k].sum()) for k in range(len(NAMES)) if bad[:, k].any()}
    Xf = np.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0); Xd = np.nan_to_num(Xd, nan=0.0, posinf=0.0, neginf=0.0)
    FEAT_F.write_bytes(pickle.dumps((Xf, yf, gf, mf, Xd, yd, gd, md, NAMES, NAN_FIX)))
if NAN_FIX: say(f"  non-finite values set to 0: {NAN_FIX}")
FAM = {}
for n in NAMES: FAM.setdefault(n.split(":")[0], []).append(n)
col = {n: k for k, n in enumerate(NAMES)}
TIMES["features"] = time.time() - t0
say(f"  {len(NAMES)} features in {len(FAM)} families: " + ", ".join(f"{k} {len(v)}" for k, v in FAM.items()))

if RUN_MAIN:
    # ---------------------------------------------------------------- 8  fusers
    say("step 7/9: fusers")
    t0 = time.time()
    import warnings; warnings.filterwarnings("ignore")
    from lightgbm import LGBMRanker, LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    NGOLD = np.array([len(gold_ids(ALLQ[i])) for i in DEVIDX])
    GI = [np.where(gd == q)[0] for q in range(len(DEVIDX))]


    def evaluate(sc):
        out = {}
        for k in (5, 10, 15, 30):
            hit = np.array([yd[ix][np.argsort(-sc[ix])[:k]].sum() for ix in GI])
            out[k] = (hit / NGOLD, (hit == NGOLD).astype(float))
        return out


    def ranker(feats_, seed=0):
        ix = [col[n] for n in feats_]
        m = LGBMRanker(objective="lambdarank", n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8,
                       subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, random_state=seed, verbose=-1)
        m.fit(Xf[:, ix], yf, group=np.bincount(gf))
        return m.predict(Xd[:, ix]), m


    def stacked(feats_):
        ix = [col[n] for n in feats_]
        m = StackingClassifier(estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=10, max_features=0.3, n_jobs=-1, random_state=0)),
            ("lgb", LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                   random_state=0, verbose=-1)),
            ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)))],
            final_estimator=LogisticRegression(max_iter=3000), cv=3, n_jobs=1).fit(Xf[:, ix], yf)
        return m.predict_proba(Xd[:, ix])[:, 1]


    PILOT12 = ["score:hyb", "score:hyb_rank", "score:dense_e1", "score:bm25_full", "score:dec", "score:dec_rank", "db:n_candidates", "db:dec_max",
               "db:dec_rank", "db:dec_minus_dbmax", "db:n_tables", "db:hyb_max"]
    RES = {}
    RES["hybrid search alone (RASL retriever-only: recall@15 92.6)"] = evaluate(Xd[:, col["score:hyb"]])
    RES["decider alone"] = evaluate(Xd[:, col["score:dec"]])
    RES[f"A  pilot's 12 features, stacked ensemble"] = evaluate(stacked(PILOT12))
    sc_rank, mC = ranker(NAMES)
    RES[f"C  all {len(NAMES)} features, lambdarank"] = evaluate(sc_rank)
    sc_stack = stacked(NAMES)
    RES[f"D  all {len(NAMES)} features, stacked ensemble"] = evaluate(sc_stack)
    zs = lambda v: np.concatenate([(v[ix] - v[ix].mean()) / (v[ix].std() + 1e-9) for ix in GI])
    order_ix = np.concatenate(GI)
    blend = np.empty(len(yd)); blend[order_ix] = zs(sc_rank) + zs(sc_stack)
    RES["E  C + D blended"] = evaluate(blend)
    for fam in FAM:
        if fam in ("score",): continue
        RES[f"   C without {fam} ({len(FAM[fam])})"] = evaluate(ranker([n for n in NAMES if n not in FAM[fam]])[0])
    TIMES["fusers"] = time.time() - t0

    # ---------------------------------------------------------------- 9  leakage audit, feature dump, report
    say("step 8/9: leakage audit and feature dump")
    auc = {}
    for n in NAMES:
        v = Xd[:, col[n]]
        if np.std(v) > 0: a = roc_auc_score(yd, v); auc[n] = max(a, 1 - a)
    sus = sorted([(a, n) for n, a in auc.items() if a > 0.98], reverse=True)
    import csv
    best_sc = blend
    dump = OUT / f"FEATURE_DUMP_{TAG}.csv"
    with open(dump, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["question_id", "question", "evidence", "table", "is_gold", "fuser_score", "fuser_rank"] + NAMES)
        for q in random.Random(5).sample(range(len(DEVIDX)), 30):
            ix = GI[q]; order = ix[np.argsort(-best_sc[ix])]
            keep = list(order[:20]) + [r for r in order[20:] if yd[r]]
            for rnk, r in enumerate(keep):
                i, j = md[r]
                w.writerow([ALLQ[i].get("question_id"), PUB[i]["question"], PUB[i]["evidence"], f"{TABLES[j][0]}.{TABLES[j][1]}", yd[r],
                            round(float(best_sc[r]), 4), int(np.where(order == r)[0][0]) + 1] + [round(float(x), 4) for x in Xd[r]])
    (OUT / f"FEATURE_LIST_{TAG}.txt").write_text("\n".join(f"{n}" for n in NAMES) + "\n")

    say("step 9/9: report")
    multi = NGOLD >= 2
    L = [f"No-LLM fuser, out of fold | BIRD dev {len(DEVIDX)} questions ({multi.sum()} need 2+ tables) | catalog {NT} tables in {len(TBL_OF)} databases",
         f"fuser fitted on {len(FITIDX)} train questions from {len(trn_dbs)} databases ({N_FOLDS} folds) | decider: {DECIDER}, loss {LOSS}, "
         f"{CAP_TRAIN_Q} questions per fold | evidence {'used' if USE_EVIDENCE else 'NOT used'} | values {'used' if USE_VALUES else 'not used'}",
         f"{len(NAMES)} features: " + ", ".join(f"{k} {len(v)}" for k, v in FAM.items()), "",
         f"{'fuser':62s} {'rec@15':>7s} {'cmp@5':>6s} {'cmp@10':>7s} {'cmp@15':>7s} {'cmp@30':>7s} {'cmp@15 2+':>9s}"]
    for k, v in RES.items():
        L.append(f"{k:62s} {100*v[15][0].mean():7.1f} {100*v[5][1].mean():6.1f} {100*v[10][1].mean():7.1f} {100*v[15][1].mean():7.1f} "
                 f"{100*v[30][1].mean():7.1f} {100*v[15][1][multi].mean():9.1f}")
    rs = np.random.default_rng(0)
    def ci(x):
        b = [x[rs.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return f"{100*x.mean():.1f} (95% CI {100*np.percentile(b, 2.5):.1f} to {100*np.percentile(b, 97.5):.1f})"
    bestname = max([k for k in RES if not k.startswith("   ")], key=lambda k: RES[k][15][0].mean())
    L += ["", f"best fuser: {bestname.strip()}", f"  recall@15 {ci(RES[bestname][15][0])}   vs RASL full (with Claude 3.5 Sonnet): 97.8, RASL retriever: 92.6",
          f"  complete@15 {ci(RES[bestname][15][1])}",
          f"  vs A, complete@15: {ci(RES[bestname][15][1] - RES[list(RES)[2]][15][1])}", "",
          "leakage audit (single features that alone separate gold from non-gold with AUC > 0.98; expected: none):"]
    L += [f"  !! {n:30s} AUC {a:.3f}" for a, n in sus] or ["  none"]
    L += [f"non-finite feature values replaced by 0: {NAN_FIX}" if NAN_FIX else "non-finite feature values: none"]
    L += ["", "strongest single features by AUC (a feature this strong on its own deserves a look):"]
    L += [f"  {n:30s} {a:.3f}" for n, a in sorted(auc.items(), key=lambda x: -x[1])[:10]]
    gain = mC.booster_.feature_importance("gain"); order = np.argsort(-gain)[:20]
    L += ["", "top 20 features of C by gain:"] + [f"  {NAMES[k]:30s} {100 * gain[k] / gain.sum():5.1f}%" for k in order]
    # one example question, so the feature values are visible here too
    q0 = GI[0]; o0 = q0[np.argsort(-best_sc[q0])][:5]; show = ["score:hyb_rank", "score:dec_rank", "score:dense_e1_z", "lex:idf_overlap",
                                                               "val:n_literals_matched", "ev:n_cols_named" if USE_EVIDENCE else "lex:fuzzy_max", "graph:dist_top1_hyb"]
    L += ["", f"example: {PUB[DEVIDX[0]]['question']}  [evidence: {PUB[DEVIDX[0]]['evidence'][:80]}]",
          f"  {'table':40s} gold " + " ".join(f"{s.split(':')[1][:14]:>14s}" for s in show)]
    L += [f"  {TABLES[md[r][1]][0] + '.' + TABLES[md[r][1]][1]:40s} {yd[r]:4d} " + " ".join(f"{Xd[r, col[s]]:14.3f}" for s in show) for r in o0]
    L += ["", f"all feature values for 30 dev questions: {dump}", "minutes: " + " | ".join(f"{k} {v / 60:.1f}" for k, v in TIMES.items())]
    (OUT / f"OOF_{TAG}.txt").write_text("\n".join(L) + "\n")
    print("\n" + "\n".join(L))

# ================================================================ EXTRAS: decider ablation, frozen scorers only, sqlglot labels, set sizes, timing
import warnings; warnings.filterwarnings("ignore")
from lightgbm import LGBMRanker, LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
rs = np.random.default_rng(0)


def fit_rank(cols_, X, y, g):
    m = LGBMRanker(objective="lambdarank", n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, random_state=0, verbose=-1)
    return m.fit(X[:, cols_], y, group=np.bincount(g))


def fit_stack(cols_, X, y):
    return StackingClassifier(estimators=[
        ("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=10, max_features=0.3, n_jobs=-1, random_state=0)),
        ("lgb", LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                               random_state=0, verbose=-1)),
        ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)))],
        final_estimator=LogisticRegression(max_iter=3000), cv=3, n_jobs=1).fit(X[:, cols_], y)


def metrics(score, y, groups_ix, ngold, ks=(5, 10, 15)):
    """Per question: recall, complete and precision when the top k candidates are kept."""
    out = {}
    for k in ks:
        hit = np.array([y[ix][np.argsort(-score[ix])[:k]].sum() for ix in groups_ix])
        out[k] = (hit / np.maximum(1, ngold), (hit >= ngold).astype(float) * (ngold > 0), hit / k)
    return out


def ci(x):
    b = [x[rs.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return f"{100*x.mean():+.1f} (95% CI {100*np.percentile(b, 2.5):+.1f} to {100*np.percentile(b, 97.5):+.1f})"


EX = []                                                                   # report lines
if RUN_EXTRAS:
    t_ex = time.time()
    GI = [np.where(gd == q)[0] for q in range(len(DEVIDX))]
    NG_RE = np.array([len(gold_ids(ALLQ[i])) for i in DEVIDX])
    # ---- 1  gold labels re-derived with sqlglot (a SQL parser) instead of the name regex
    say("extras 1/5: gold labels with sqlglot")
    import sqlglot
    from sqlglot import exp

    def sg_tables(q):
        try:
            tree = sqlglot.parse_one(q["SQL"], read="sqlite")
            names = {t.name.lower() for t in tree.find_all(exp.Table)} - {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
        except Exception:
            return None
        return {i for i in TBL_OF[q["db_id"]] if TABLES[i][1].lower() in names}

    SG, n_fail, diffs = [], 0, []
    for i in DEVIDX:
        q = ALLQ[i]; re_set = set(gold_ids(q)); sg = sg_tables(q)
        if sg is None: n_fail += 1; sg = re_set
        if sg != re_set: diffs.append((q.get("question_id"), sorted(TABLES[j][1] for j in re_set), sorted(TABLES[j][1] for j in sg)))
        SG.append(sg)
    pos = {i: k for k, i in enumerate(DEVIDX)}
    y_sg = np.array([int(j in SG[pos[i]]) for i, j in md]); NG_SG = np.array([len(s) for s in SG])
    EX += ["== 1  gold labels: name regex vs sqlglot parser (dev)",
           f"same table set: {len(DEVIDX) - len(diffs)} / {len(DEVIDX)} questions | different: {len(diffs)} | sqlglot could not parse: {n_fail} (regex kept)"]
    EX += [f"  question {a}: regex {b} | sqlglot {c}" for a, b, c in diffs[:10]]

    # ---- 2  feature sets: all, without the decider, frozen scorers only, hand-engineered only
    say("extras 2/5: fusers without the decider and with frozen scorers only (stacked ensembles take ~12 minutes each)")
    rank_cols = [n for n in NAMES if n.startswith("score:") and n.endswith("_rank") and not n.startswith("score:dec")]
    def agree_nd(X):
        R = X[:, [col[n] for n in rank_cols]]
        return np.column_stack([R.min(1), R.max(1), R.mean(1), R.std(1), (R < 5).sum(1), (R < 15).sum(1)])
    AGND = ["agree_nd:rank_min", "agree_nd:rank_max", "agree_nd:rank_mean", "agree_nd:rank_std", "agree_nd:n_top5", "agree_nd:n_top15"]
    Xf2, Xd2 = np.hstack([Xf, agree_nd(Xf)]), np.hstack([Xd, agree_nd(Xd)]); N2 = NAMES + AGND; c2 = {n: k for k, n in enumerate(N2)}
    WITH_DEC = {"agree:rank_min", "agree:rank_max", "agree:rank_mean", "agree:rank_std", "agree:n_top5", "agree:n_top15", "agree:dec_vs_hyb",
                "graph:dist_top1_dec", "graph:n_nbr_top5_dec", "graph:bridge"}
    SETS = {"all features (with decider)": NAMES,
            "no decider (every decider-derived feature removed)": [n for n in N2 if not (n.startswith(("score:dec", "db:dec")) or n in WITH_DEC)]}
    SETS["frozen scorers only (scores, agreement, database; no hand-made features)"] = [n for n in SETS["no decider (every decider-derived feature removed)"]
                                                                                       if n.split(":")[0] in ("score", "agree", "agree_nd", "db")]
    SETS["hand-made features only (no retrieval scores)"] = [n for n in SETS["no decider (every decider-derived feature removed)"]
                                                             if n.split(":")[0] in ("lex", "cov", "ev", "val", "type", "meta", "graph", "q", "emb")]
    SCORES, MODELS = {}, {}
    SCORES["hybrid search alone"] = Xd[:, col["score:hyb"]]
    SCORES["decider alone"] = Xd[:, col["score:dec"]]
    for nm, fs in SETS.items():
        ix = [c2[n] for n in fs]
        SCORES[f"C lambdarank | {nm} ({len(fs)})"] = fit_rank(ix, Xf2, yf, gf).predict(Xd2[:, ix])
        if nm.startswith(("all", "no decider")):
            say(f"  stacked ensemble: {nm}")
            MODELS[nm] = m = fit_stack(ix, Xf2, yf)
            SCORES[f"D stacked  | {nm} ({len(fs)})"] = m.predict_proba(Xd2[:, ix])[:, 1]
    for lab, y_, ng in (("name-regex labels (as in the main run)", yd, NG_RE), ("sqlglot labels", y_sg, NG_SG)):
        EX += ["", f"== 2  fusers, BIRD dev {len(DEVIDX)} questions, {lab}",
               f"{'fuser':92s} {'R@5':>5s} {'R@10':>5s} {'R@15':>5s} {'C@5':>5s} {'C@10':>5s} {'C@15':>5s}"]
        MM = {k: metrics(v, y_, GI, ng) for k, v in SCORES.items()}
        for k, m in MM.items():
            EX.append(f"{k:92s} " + " ".join(f"{100*m[kk][0].mean():5.1f}" for kk in (5, 10, 15)) + " "
                      + " ".join(f"{100*m[kk][1].mean():5.1f}" for kk in (5, 10, 15)))
        kd = [k for k in MM if k.startswith("D stacked  | all")][0]; kn = [k for k in MM if k.startswith("D stacked  | no decider")][0]
        EX += [f"decider's contribution (D all minus D no decider): recall@15 {ci(MM[kd][15][0] - MM[kn][15][0])}, "
               f"complete@15 {ci(MM[kd][15][1] - MM[kn][15][1])}, complete@5 {ci(MM[kd][5][1] - MM[kn][5][1])}",
               f"RASL full (Claude 3.5 Sonnet) recall@15 97.8 | D all recall@15 {100*MM[kd][15][0].mean():.1f} | D no decider {100*MM[kn][15][0].mean():.1f}"]

    # ---- 3  set sizes: fixed k, and an adaptive set (keep tables above a probability threshold, 1 to 15 tables)
    say("extras 3/5: set sizes")
    p_all = SCORES[[k for k in SCORES if k.startswith("D stacked  | all")][0]]
    EX += ["", "== 3  set size vs recall (D, all features, sqlglot labels)", f"{'set':34s} {'avg tables':>10s} {'recall':>7s} {'complete':>9s} {'precision':>9s}"]
    for k in (1, 2, 3, 5, 10, 15):
        m = metrics(p_all, y_sg, GI, NG_SG, ks=(k,))[k]
        EX.append(f"{'top ' + str(k):34s} {k:10.1f} {100*m[0].mean():7.1f} {100*m[1].mean():9.1f} {100*m[2].mean():9.1f}")
    for tau in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02):
        sz, rc, cp, pr = [], [], [], []
        for q, ix in enumerate(GI):
            o = ix[np.argsort(-p_all[ix])]; keep = [r for r in o if p_all[r] >= tau][:15] or [o[0]]
            h = y_sg[keep].sum(); sz.append(len(keep)); rc.append(h / max(1, NG_SG[q])); cp.append(float(h >= NG_SG[q])); pr.append(h / len(keep))
        EX.append(f"{'probability >= ' + str(tau) + ' (1 to 15)':34s} {np.mean(sz):10.1f} {100*np.mean(rc):7.1f} {100*np.mean(cp):9.1f} {100*np.mean(pr):9.1f}")

    # ---- 4  timing per query (one question at a time, batch size 1)
    say("extras 4/5: timing per query")
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    e1 = SentenceTransformer(EMB1, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    e2 = SentenceTransformer(EMB2, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    T1 = e1.encode(CARD, normalize_embeddings=True); T1n = e1.encode(NAMECARD, normalize_embeddings=True); T2 = e2.encode(CARD, normalize_embeddings=True)
    CEc = e1.encode([c[1] for c in COLTXT], normalize_embeddings=True); COLT = np.array([c[0] for c in COLTXT])
    b1, b2 = BM25Okapi([words(c) for c in CARD]), BM25Okapi([words(c) for c in NAMECARD])
    dpath = OUT / f"decider_{TAG}_dev"
    dm = AutoModelForSequenceClassification.from_pretrained(str(dpath), num_labels=1, trust_remote_code=True).cuda().eval()
    dt = AutoTokenizer.from_pretrained(str(dpath), trust_remote_code=True)
    pca = PCA(16, random_state=0).fit((SC["qe"][[m_[0] for m_ in mf]].astype(np.float32) * SC["te"][[m_[1] for m_ in mf]].astype(np.float32))[::7])
    fuser = MODELS["all features (with decider)"]; sync = torch.cuda.synchronize
    TT = {k: [] for k in ("1 embed question (2 models)", "2 BM25 (2 views)", "3 dense scores + best column", "4 hybrid top 100",
                          "5 decider (100 pairs)", "6 features (139)", "7 fuser")}
    for i in DEVIDX[:100] + DEVIDX[:5]:
        p = PUB[i]; t = time.perf_counter()
        q1 = e1.encode([qtext(p)], prompt_name="query", normalize_embeddings=True)
        q2 = e2.encode([qtext(p)], prompt="Represent this sentence for searching relevant passages: ", normalize_embeddings=True); sync()
        t1 = time.perf_counter(); s_b1 = b1.get_scores(words(qtext(p))); s_b2 = b2.get_scores(words(qtext(p))); t2 = time.perf_counter()
        d1 = q1 @ T1.T; d1n = q1 @ T1n.T; d2 = q2 @ T2.T; qc = (q1 @ CEc.T)[0]
        bc = np.full(NT, -1.0); np.maximum.at(bc, COLT, qc); t3 = time.perf_counter()
        r = lambda v: np.argsort(np.argsort(-v)); hy = 1 / (60 + r(d1[0])) + 1 / (60 + r(s_b1)); cand = np.argsort(-hy)[:K_CAND]; t4 = time.perf_counter()
        x = dt([qtext(p)] * K_CAND, [CARD[j] for j in cand], truncation=True, max_length=320, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): lg = dm(**x).logits[:, 0].float().cpu().numpy()
        sync(); t5 = time.perf_counter()
        rows = feats(i); xr = np.array([[rw[k] for k in NAMES[:-16]] for rw in rows], dtype=np.float32)
        qt = SC["qe"][i].astype(np.float32)[None, :] * SC["te"][CAND[i]].astype(np.float32)
        xr = np.nan_to_num(np.hstack([xr, pca.transform(qt)])); t6 = time.perf_counter()
        fuser.predict_proba(xr); t7 = time.perf_counter()
        for k, v in zip(TT, (t1 - t, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6)): TT[k].append(1000 * v)
    TT = {k: v[5:] for k, v in TT.items()}                                # the first 5 calls are warm-up
    EX += ["", f"== 4  latency per question, batch size 1, {torch.cuda.get_device_name(0)} (median and 90th percentile, ms)"]
    EX += [f"  {k:36s} {np.median(v):8.1f} {np.percentile(v, 90):8.1f}" for k, v in TT.items()]
    tot = np.sum([np.array(v) for v in TT.values()], 0)
    EX += [f"  {'total':36s} {np.median(tot):8.1f} {np.percentile(tot, 90):8.1f}",
           "  GPU models: Qwen3-Embedding-0.6B, BGE-large (0.34B), OpenDecider-nano (0.4B, fine-tuned). Fuser and features on CPU. No LLM call."]
    del dm; gc.collect(); torch.cuda.empty_cache()
    TIMES["extras (BIRD)"] = time.time() - t_ex

# ================================================================ BEAVER: zero-shot transfer (deciders and fuser trained on BIRD only)
if RUN_BEAVER:
    import pandas as pd
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    t_bv = time.time()
    say("BEAVER 1/4: loading the current release")
    findf = lambda name: (glob.glob(f"{BEAVER_DIR}/**/{name}", recursive=True) or [None])[0]
    if not findf("table_dw.parquet"):
        EX += ["", f"== 5  BEAVER: no table_dw.parquet under {BEAVER_DIR}; skipped (run the BEAVER download cell first)"]
    else:
        # features that do not exist on BEAVER (no evidence, no cell values, no column types, no row counts) are left out of this fuser
        TRANSFER = [n for n in NAMES if n.split(":")[0] not in ("ev", "val", "type") and n not in ("meta:log_rows", "q:has_evidence")]
        tix = [col[n] for n in TRANSFER]
        say(f"  fitting the transfer fuser on BIRD ({len(TRANSFER)} features)")
        bC = fit_rank(tix, Xf, yf, gf); bD = fit_stack(tix, Xf, yf)
        if "pca" not in globals():
            pca = PCA(16, random_state=0).fit((SC["qe"][[m_[0] for m_ in mf]].astype(np.float32) * SC["te"][[m_[1] for m_ in mf]].astype(np.float32))[::7])
        KEYS = json.loads(Path(KEYS_FILE).read_text()) if Path(KEYS_FILE).exists() else {}
        bt, bc = [], []
        for db in ("dw", "nova", "neutron"):
            for _, r in pd.read_parquet(findf(f"table_{db}.parquet")).iterrows():
                cs = r["column_names"]; cs = json.loads(cs) if isinstance(cs, str) else list(cs)
                bt.append((db, str(r["table_name"]))); bc.append([str(c) for c in cs])
        # the catalog globals that feats() reads are rebound to BEAVER
        TABLES, NT = bt, len(bt)
        CARDT = None                                                      # set below, after CARD
        COLS = [[(c, c.replace("_", " ").lower()) for c in cs] for cs in bc]
        CARD = [f"{t} (database {db}): " + ", ".join(c for c, _ in COLS[k]) for k, (db, t) in enumerate(TABLES)]
        NAMECARD = [f"{camel(t).replace('_', ' ')} (database {db})" for db, t in TABLES]
        CARDT = CARD                                                      # the decider reads CARDT
        NCOL = [len(c) for c in COLS]; CTYPE = [["text"] * n for n in NCOL]; NPK = [0] * NT
        IDX = {(db, t.upper()): k for k, (db, t) in enumerate(TABLES)}
        ADJ = {k: set() for k in range(NT)}; fkcols = {k: set() for k in range(NT)}
        for k, (db, t) in enumerate(TABLES):
            for a, ref, b in KEYS.get(db, {}).get(t.upper(), {}).get("fk", []):
                o = IDX.get((db, str(ref).upper()))
                if o is not None and o != k: ADJ[k].add(o); ADJ[o].add(k); fkcols[k] |= set(a)
        NFKC = [len(fkcols[k]) for k in range(NT)]
        COLTXT = [(k, f"{t} . {c} ({nat})") for k, (_, t) in enumerate(TABLES) for c, nat in COLS[k]]
        TBL_OF = {}
        for k, (d, t) in enumerate(TABLES): TBL_OF.setdefault(d, []).append(k)
        TTOK = [set(toks(t)) for _, t in TABLES]
        CTOK = [set(toks(t) + [w for c, nat in COLS[k] for w in toks(c) + toks(nat)]) for k, (_, t) in enumerate(TABLES)]
        CNAMES = [{c.lower() for c, _ in COLS[k]} - {"id", "name", "type", "date", "year", "description", "code"} for k in range(NT)]
        DF = {}
        for ts in CTOK:
            for w in ts: DF[w] = DF.get(w, 0) + 1
        IDF = {w: np.log(NT / (1 + d)) for w, d in DF.items()}; VOCAB = sorted(DF); VID = {w: k for k, w in enumerate(VOCAB)}
        PR = np.zeros(NT)
        for d, ids in TBL_OF.items():
            n = len(ids); loc = {j: k for k, j in enumerate(ids)}; r = np.ones(n) / n
            for _ in range(30):
                nr = np.full(n, 0.15 / n)
                for j in ids:
                    nb = ADJ[j]
                    for b_ in nb: nr[loc[b_]] += 0.85 * r[loc[j]] / len(nb)
                    if not nb: nr += 0.85 * r[loc[j]] / n
                r = nr
            for j in ids: PR[j] = r[loc[j]] * n
        VALS, NUMR, NROWS, VCOLS, NCOLS_R = {}, {}, {}, {}, {}
        # questions: a sample of the synthetic questions and all real ones
        BQ = []
        for sp in ("dw", "nova", "neutron", "dw_real"):
            f_ = findf(f"query_{sp}.parquet")
            if not f_: continue
            for _, r in pd.read_parquet(f_).iterrows():
                tabs = r["tables"]; tabs = json.loads(tabs) if isinstance(tabs, str) else list(tabs)
                g = {IDX.get((str(r["db"]), str(x).upper())) for x in tabs}
                if tabs and None not in g: BQ.append(dict(question=str(r["question"]), gold=g, real=sp == "dw_real"))
        syn = [q for q in BQ if not q["real"]]; real = [q for q in BQ if q["real"]]
        BQ = random.Random(7).sample(syn, min(1000, len(syn))) + real
        PUB = [{"question": q["question"], "evidence": ""} for q in BQ]
        say(f"BEAVER 2/4: {NT} tables, {len(syn)} synthetic questions (1,000 sampled) and {len(real)} real; scorers")
        SC = {}
        for nm, docs in (("bm25_full", CARD), ("bm25_name", NAMECARD)):
            bb = BM25Okapi([words(c) for c in docs]); SC[nm] = np.stack([bb.get_scores(words(qtext(p))) for p in PUB]).astype(np.float32)
        for tag, mname in (("e1", EMB1), ("e2", EMB2)):
            emb = SentenceTransformer(mname, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
            kw = {"prompt_name": "query"} if tag == "e1" else {"prompt": "Represent this sentence for searching relevant passages: "}
            QE = emb.encode([qtext(p) for p in PUB], normalize_embeddings=True, batch_size=64, **kw)
            TE = emb.encode(CARD, normalize_embeddings=True, batch_size=16)
            SC[f"dense_{tag}_full"] = (QE @ TE.T).astype(np.float32)
            if tag == "e1":
                SC["dense_e1_name"] = (QE @ emb.encode(NAMECARD, normalize_embeddings=True).T).astype(np.float32)
                CE = emb.encode([c[1] for c in COLTXT], normalize_embeddings=True, batch_size=64); COLT = np.array([c[0] for c in COLTXT])
                QC = QE @ CE.T; bcm = np.full((len(QE), NT), -1.0, dtype=np.float32)
                for j in range(NT):
                    m_ = COLT == j
                    if m_.any(): bcm[:, j] = QC[:, m_].max(1)
                SC["dense_e1_bestcol"] = bcm; SC["qe"] = QE.astype(np.float16); SC["te"] = TE.astype(np.float16); del QC, CE
            del emb; gc.collect(); torch.cuda.empty_cache()
        K_CAND = min(K_CAND, NT)
        HYB = 1 / (60 + rankm(SC["dense_e1_full"])) + 1 / (60 + rankm(SC["bm25_full"])); CAND = np.argsort(-HYB, 1)[:, :K_CAND]
        say("BEAVER 3/4: decider (trained on BIRD) scores the candidates")
        sc_ = score(OUT / f"decider_{TAG}_dev", list(range(len(BQ)))); DEC = np.stack([sc_[i] for i in range(len(BQ))])
        say("BEAVER 4/4: features and the transfer fuser")
        Xb, yb, gb = [], [], []
        for i in range(len(BQ)):
            rows = feats(i)
            xr = np.array([[rw[k] for k in NAMES[:-16]] for rw in rows], dtype=np.float32)
            qt = SC["qe"][i].astype(np.float32)[None, :] * SC["te"][CAND[i]].astype(np.float32)
            Xb.append(np.hstack([xr, pca.transform(qt)])); yb += [int(j in BQ[i]["gold"]) for j in CAND[i]]; gb += [i] * K_CAND
        Xb = np.nan_to_num(np.vstack(Xb)); yb, gb = np.array(yb), np.array(gb)
        SB = {"hybrid search alone": Xb[:, col["score:hyb"]], "decider alone (BIRD-trained)": Xb[:, col["score:dec"]],
              f"C lambdarank, BIRD-fitted ({len(TRANSFER)} features)": bC.predict(Xb[:, tix]),
              f"D stacked, BIRD-fitted ({len(TRANSFER)} features)": bD.predict_proba(Xb[:, tix])[:, 1]}
        for lab, sel in (("1,000 synthetic questions", [i for i, q in enumerate(BQ) if not q["real"]]), (f"{len(real)} real questions", [i for i, q in enumerate(BQ) if q["real"]])):
            GIb = [np.where(gb == i)[0] for i in sel]; NGb = np.array([len(BQ[i]["gold"]) for i in sel])
            EX += ["", f"== 5  BEAVER (current release, {NT} tables pooled), {lab}, zero-shot: nothing trained on BEAVER",
                   f"{'fuser':52s} {'R@5':>5s} {'R@10':>5s} {'R@15':>5s} {'C@5':>5s} {'C@8':>5s} {'C@15':>5s} {'C@25':>5s}"]
            for k, v in SB.items():
                m = metrics(v, yb, GIb, NGb, ks=(5, 8, 10, 15, 25))
                EX.append(f"{k:52s} " + " ".join(f"{100*m[kk][0].mean():5.1f}" for kk in (5, 10, 15)) + " "
                          + " ".join(f"{100*m[kk][1].mean():5.1f}" for kk in (5, 8, 15, 25)))
        EX += ["paper 1 on the 121 real questions (trained on BEAVER synthetic questions): search R@15 74.3 C@15 42.1 | NIFT-SQL R@15 80.8 C@15 54.5"]
    TIMES["BEAVER"] = time.time() - t_bv

if EX:
    EX = [f"Extras | TAG {TAG} | {time.strftime('%Y-%m-%d %H:%M')}", ""] + EX + ["", "minutes: " + " | ".join(f"{k} {v / 60:.1f}" for k, v in TIMES.items())]
    (OUT / f"EXTRAS_{TAG}.txt").write_text("\n".join(EX) + "\n")
    print("\n" + "\n".join(EX))

# ================================================================ SELECT: which search builds the candidates, which scorers the fuser keeps
# Rules fixed before running. All choices are made by cross-validation on the training questions (5 folds by database);
# BIRD dev is printed for information only and is not used to choose.
#   candidates: the search combination with the highest complete@100 (sqlglot labels); within 0.2 points, fewer scorers wins.
#   scorers:    remove each scorer in turn (its score features, its agreement terms); keep it only if removing it costs
#               >= 0.3 points of complete@5 or complete@15. The decider is left out (it is reported as an ablation).
if RUN_SELECT:
    t_se = time.time()
    import sqlglot
    from sqlglot import exp
    say("select 1/3: sqlglot gold labels for the training and dev questions")

    def sg_ids(q):
        try:
            tree = sqlglot.parse_one(q["SQL"], read="sqlite")
            names = {t.name.lower() for t in tree.find_all(exp.Table)} - {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
            return {i for i in TBL_OF[q["db_id"]] if TABLES[i][1].lower() in names}
        except Exception:
            return set(gold_ids(q))

    GOLD = {i: sg_ids(ALLQ[i]) for i in FITIDX + DEVIDX}
    SE = [f"Scorer selection | TAG {TAG} | {time.strftime('%Y-%m-%d %H:%M')} | sqlglot labels | choices by 5-fold CV on {len(FITIDX)} training questions", ""]

    # ---- 1  candidate pool: reciprocal rank fusion of scorer subsets, over the whole catalog
    say("select 2/3: candidate pool options")
    RK = {k: np.argsort(np.argsort(-SC[v], 1), 1) for k, v in (("bm25", "bm25_full"), ("bm25_name", "bm25_name"), ("qwen3", "dense_e1_full"),
                                                                 ("qwen3_name", "dense_e1_name"), ("qwen3_bestcol", "dense_e1_bestcol"), ("bge", "dense_e2_full"))}
    OPTS = [("bm25",), ("qwen3",), ("bge",), ("bm25", "qwen3"), ("bm25", "bge"), ("qwen3", "bge"), ("bm25", "qwen3", "bge"),
            ("bm25", "bm25_name", "qwen3", "qwen3_name", "qwen3_bestcol", "bge")]

    def pool(opt, idx, k=100):
        S_ = sum(1 / (60 + RK[o][idx]) for o in opt)
        top = np.argsort(-S_, 1)[:, :k]
        r = [len(GOLD[i] & set(t)) / max(1, len(GOLD[i])) for i, t in zip(idx, top)]
        c = [float(GOLD[i] <= set(t)) for i, t in zip(idx, top)]
        return 100 * np.mean(r), 100 * np.mean(c)

    fi, di = np.array(FITIDX), np.array(DEVIDX)
    SE += ["== 1  candidate pool (top 100 by reciprocal rank fusion)", f"{'scorers fused':52s} {'train R@100':>11s} {'train C@100':>11s} {'dev R@100':>9s} {'dev C@100':>9s}"]
    PO = {}
    for o in OPTS:
        PO[o] = pool(o, fi); dv = pool(o, di)
        SE.append(f"{' + '.join(o):52s} {PO[o][0]:11.1f} {PO[o][1]:11.1f} {dv[0]:9.1f} {dv[1]:9.1f}")
    best_c = max(v[1] for v in PO.values())
    win = min([o for o in OPTS if PO[o][1] >= best_c - 0.2], key=len)
    SE += [f"chosen by the rule: {' + '.join(win)} (current pipeline uses bm25 + qwen3)"]
    if set(win) != {"bm25", "qwen3"}:
        SE += ["NOTE: the chosen pool differs from the current one, so the frozen run must rebuild candidates and features with it."]

    # ---- 2  drop one scorer at a time from the fuser (no decider), 5-fold CV on the training questions
    say("select 3/3: drop-one-scorer cross-validation (lambdarank, about 15 minutes)")
    yfs = np.array([int(j in GOLD[i]) for i, j in mf]); yds = np.array([int(j in GOLD[i]) for i, j in md])
    SCN = {"bm25": "bm25_full", "bm25_name": "bm25_name", "qwen3": "dense_e1", "qwen3_name": "dense_e1_name", "qwen3_bestcol": "bestcol", "bge": "dense_e2"}
    PAIRS = {"agree:dense_vs_bm25": ("qwen3", "bm25"), "agree:e1_vs_e2": ("qwen3", "bge"), "agree:name_vs_full": ("qwen3_name", "qwen3")}
    DECF = {"agree:rank_min", "agree:rank_max", "agree:rank_mean", "agree:rank_std", "agree:n_top5", "agree:n_top15", "agree:dec_vs_hyb",
            "graph:dist_top1_dec", "graph:n_nbr_top5_dec", "graph:bridge"}
    BASE_F = [n for n in NAMES if not (n.startswith(("score:dec", "db:dec")) or n in DECF)]

    def featset(drop=None):
        """Feature columns without the decider and without scorer `drop`, plus agreement recomputed from the scorers kept."""
        keep_sc = [s_ for s_ in SCN if s_ != drop]
        own = {f"score:{SCN[drop]}"} | {f"score:{SCN[drop]}_{t}" for t in ("rank", "z", "gap_best", "pctl")} if drop else set()
        fs = [n for n in BASE_F if not (n in own or (drop == "qwen3" and n.startswith("db:dense_e1")) or (drop and n in PAIRS and drop in PAIRS[n]))]
        rc = [col[f"score:{SCN[s_]}_rank"] for s_ in keep_sc] + [col["score:hyb_rank"]]
        def agree(X):
            R_ = X[:, rc]
            return np.column_stack([R_.min(1), R_.max(1), R_.mean(1), R_.std(1), (R_ < 5).sum(1), (R_ < 15).sum(1)])
        return [col[n] for n in fs], agree

    def build_X(X, cols_, agree):
        return np.hstack([X[:, cols_], agree(X)])

    fold_of_q = np.array([FOLD[ALLQ[i]["db_id"]] for i in FITIDX])
    row_fold = fold_of_q[gf]
    def cv(cols_, agree):
        Xa = build_X(Xf, cols_, agree); sc_ = np.zeros(len(yfs))
        for k in range(N_FOLDS):
            tr, te = row_fold != k, row_fold == k
            m_ = LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8,
                            subsample_freq=1, colsample_bytree=0.7, random_state=0, verbose=-1)
            m_.fit(Xa[tr], yfs[tr], group=np.bincount(gf[tr])[np.unique(gf[tr])])
            sc_[te] = m_.predict(Xa[te])
        GIf = [np.where(gf == q)[0] for q in range(len(FITIDX))]; NGf = np.array([len(GOLD[i]) for i in FITIDX])
        mt = metrics(sc_, yfs, GIf, NGf)
        m_ = LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8,
                        subsample_freq=1, colsample_bytree=0.7, random_state=0, verbose=-1).fit(Xa, yfs, group=np.bincount(gf))
        GId = [np.where(gd == q)[0] for q in range(len(DEVIDX))]; NGd = np.array([len(GOLD[i]) for i in DEVIDX])
        md_ = metrics(m_.predict(build_X(Xd, cols_, agree)), yds, GId, NGd)
        return mt, md_

    full = cv(*featset())
    SE += ["", "== 2  drop one scorer from the fuser (no decider; lambdarank; complete@5 / complete@15 / recall@15)",
           f"{'fuser':40s} {'CV C@5':>7s} {'CV C@15':>7s} {'CV R@15':>7s}   {'dev C@5':>7s} {'dev C@15':>8s} {'dev R@15':>8s}   decision"]
    line = lambda nm, r, dec: (f"{nm:40s} {100*r[0][5][1].mean():7.1f} {100*r[0][15][1].mean():7.1f} {100*r[0][15][0].mean():7.1f}   "
                               f"{100*r[1][5][1].mean():7.1f} {100*r[1][15][1].mean():8.1f} {100*r[1][15][0].mean():8.1f}   {dec}")
    SE.append(line("all scorers", full, ""))
    KEEP, COST = [], {}
    for s_ in SCN:
        r = cv(*featset(s_)); say(f"  without {s_} done")
        cost5 = 100 * (full[0][5][1].mean() - r[0][5][1].mean()); cost15 = 100 * (full[0][15][1].mean() - r[0][15][1].mean())
        keep = cost5 >= 0.3 or cost15 >= 0.3; COST[s_] = max(cost5, cost15)
        if keep: KEEP.append(s_)
        SE.append(line(f"without {s_}", r, f"{'KEEP' if keep else 'drop'} (costs {cost5:+.1f} C@5, {cost15:+.1f} C@15)"))
    # redundant scorers can each look free to drop but not together: drop them all at once and check; add back while it costs >= 0.3

    def featset_multi(drops):
        cols_, ag = featset(None); names_keep = [NAMES[c] for c in cols_]
        for d_ in drops:
            c2_, _ = featset(d_); names_keep = [n for n in names_keep if n in {NAMES[c] for c in c2_}]
        keep_sc = [s_ for s_ in SCN if s_ not in drops]
        rc = [col[f"score:{SCN[s_]}_rank"] for s_ in keep_sc] + [col["score:hyb_rank"]]
        def agree(X):
            R_ = X[:, rc]
            return np.column_stack([R_.min(1), R_.max(1), R_.mean(1), R_.std(1), (R_ < 5).sum(1), (R_ < 15).sum(1)])
        return [col[n] for n in names_keep], agree

    drops = [s_ for s_ in SCN if s_ not in KEEP]
    while drops:
        r = cv(*featset_multi(drops))
        cost5 = 100 * (full[0][5][1].mean() - r[0][5][1].mean()); cost15 = 100 * (full[0][15][1].mean() - r[0][15][1].mean())
        SE.append(line(f"without {', '.join(drops)}", r, f"together: costs {cost5:+.1f} C@5, {cost15:+.1f} C@15"))
        if cost5 < 0.3 and cost15 < 0.3: break
        back = max(drops, key=lambda d_: COST[d_]); drops.remove(back); KEEP.append(back)
        SE.append(f"{'':40s} -> adding back {back}")
    SE += ["", f"scorers kept by the rule: {', '.join(KEEP) if KEEP else 'none beyond the hybrid score'}",
           "(the hybrid score itself and every hand-made feature family stay; the decider stays out)"]
    TIMES["selection"] = time.time() - t_se
    (OUT / f"SELECT_{TAG}.txt").write_text("\n".join(SE) + "\n")
    print("\n" + "\n".join(SE))
