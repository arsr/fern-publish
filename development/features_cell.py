# ============ ONE CELL: directional test. Does feature engineering + a ranking objective make the fuser better? No LLM. A100 40GB is fine. ============
# Reuses the decider the pilot fine-tuned. Candidates = hybrid search top 100 (the pilot showed it holds 99.5% of gold tables).
# Compares, on all BIRD dev questions:
#   A  pilot fuser: 12 score features, stacked classifier (RF + LightGBM + LR), fitted on 150 questions
#   B  same 12 features, LightGBM lambdarank (a ranking objective), fitted on every question of the held-out train databases
#   C  engineered features (54, in 7 families), lambdarank, same fitting questions
#   C minus one family at a time (what each family is worth)
# About 15 minutes. Results go to OUT_DIR/FEATURES.txt.
PILOT_DIR = "/content/drive/MyDrive/nift_rasl/pilot"
PILOT_TAG = "t1000_v150_d100_bce"        # pilot run whose fine-tuned decider is reused
N_TRAIN_Q, N_VAL_Q = 1000, 150           # must match that pilot run (they define which databases the decider saw)
N_DEV_Q = 0                              # 0 = all 1,534 dev questions
K_CAND = 100
EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"
# =====================================================================================================================
import os, subprocess, sys, time
T0 = time.time()
def say(msg): print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)
say("step 1/7: installing (1 to 2 minutes)")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rank_bm25", "sentence-transformers", "lightgbm", "remotezip", "rapidfuzz"], check=True)
from google.colab import drive
drive.mount("/content/drive")
import json, re, zipfile, random, glob
from pathlib import Path
import numpy as np
import torch, gc
PILOT = Path(PILOT_DIR); OUT = PILOT / "features_test"; OUT.mkdir(parents=True, exist_ok=True)
DEC_DIR = PILOT / f"decider_{PILOT_TAG}"
assert (DEC_DIR / "config.json").exists(), f"no fine-tuned decider at {DEC_DIR}; run the pilot first"

# ---------------------------------------------------------------- 2  BIRD json (no sqlite needed here)
say("step 2/7: BIRD questions and schemas")
DEV = Path("/content/bird_dev"); TRN = Path("/content/bird_train"); DEV.mkdir(exist_ok=True); TRN.mkdir(exist_ok=True)
if not (DEV / "dev.json").exists():
    subprocess.run(["wget", "-q", "--show-progress", "--progress=bar:force:noscroll", "-O", "/content/dev.zip",
                    "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"], check=True)
    with zipfile.ZipFile("/content/dev.zip") as z:
        for want in ("dev.json", "dev_tables.json"):
            (DEV / want).write_bytes(z.read([n for n in z.namelist() if n.endswith("/" + want) or n == want][0]))
if not (TRN / "train.json").exists():
    from remotezip import RemoteZip
    with RemoteZip("https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip") as z:
        for want in ("train.json", "train_tables.json"):
            (TRN / want).write_bytes(z.read([n for n in z.namelist() if n.endswith("/" + want) or n == want][0]))
DEV_Q_ALL = json.loads((DEV / "dev.json").read_text())
TRN_Q_ALL = json.loads((TRN / "train.json").read_text())
SCHEMAS = json.loads((TRN / "train_tables.json").read_text()) + json.loads((DEV / "dev_tables.json").read_text())

# ---------------------------------------------------------------- 3  catalog (same cards as the pilot) + columns + foreign-key graph
camel = lambda s: re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
STOP = set("the a an of in on for to and or is are was were be by with what which who whom whose how many much list give show find "
           "name names all each per than that this these those there their its it as at from among between please tell me do does did "
           "have has had more most less least number".split())
def toks(s):
    out = []
    for w in re.findall(r"[a-z0-9]+", camel(s).lower().replace("_", " ")):
        if w in STOP: continue
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out

TABLES, CARD, COLS, COLTXT, NCOL = [], [], [], [], []
ADJ = {}
for s in SCHEMAS:
    db, tn = s["db_id"], s["table_names_original"]
    cid = {i: (t, c) for i, (t, c) in enumerate(s["column_names_original"])}
    base = len(TABLES)
    for ti, t in enumerate(tn):
        mine = [(c, nat) for (tt, c), (_, nat) in zip(s["column_names_original"], s["column_names"]) if tt == ti]
        desc = [c if nat.lower().replace(" ", "") == c.lower().replace("_", "") else f"{c} ({nat})" for c, nat in mine]
        TABLES.append((db, t)); CARD.append(f"{t} (database {db.replace('_', ' ')}): " + ", ".join(desc))
        COLS.append(mine); NCOL.append(len(mine))
        for c, nat in mine: COLTXT.append((len(TABLES) - 1, f"{t} . {c} ({nat})", c, nat))
        ADJ[len(TABLES) - 1] = set()
    for a, b in s["foreign_keys"]:
        ta, tb = base + cid[a][0], base + cid[b][0]
        if ta != tb: ADJ[ta].add(tb); ADJ[tb].add(ta)
TBL_OF = {}
for i, (d, t) in enumerate(TABLES): TBL_OF.setdefault(d, []).append(i)
TTOK = [set(toks(t)) for _, t in TABLES]
CTOK = [set(toks(t) + [w for c, nat in COLS[i] for w in toks(c) + toks(nat)]) for i, (_, t) in enumerate(TABLES)]
DF = {}
for ts in CTOK:
    for w in ts: DF[w] = DF.get(w, 0) + 1
IDF = {w: np.log(len(TABLES) / (1 + d)) for w, d in DF.items()}
VOCAB = sorted(DF); VID = {w: i for i, w in enumerate(VOCAB)}
TOK_TABLES = [[] for _ in VOCAB]
for i, ts in enumerate(CTOK):
    for w in ts: TOK_TABLES[VID[w]].append(i)
print(f"catalog: {len(TBL_OF)} databases, {len(TABLES)} tables, {len(COLTXT)} columns")


def gold_ids(q):
    sql = q["SQL"].lower()
    return [i for i in TBL_OF[q["db_id"]] if re.search(r'(?<![\w])[`"\[]?' + re.escape(TABLES[i][1].lower()) + r'[`"\]]?(?![\w])', sql)]


qtext = lambda q: (q["question"] + " " + q.get("evidence", "")).strip()
# the same split as the pilot: the decider was trained on FIT databases only
rng = random.Random(0)
trn_dbs = sorted({q["db_id"] for q in TRN_Q_ALL}); rng.shuffle(trn_dbs)
VAL_DBS = set(trn_dbs[:max(5, len(trn_dbs) // 10)])
valpool = [q for q in TRN_Q_ALL if q["db_id"] in VAL_DBS and gold_ids(q)]
VAL150 = random.Random(2).sample(valpool, min(N_VAL_Q, len(valpool)))
FITQ = valpool                                                           # every question on the held-out train databases
DEVQ = random.Random(3).sample(DEV_Q_ALL, N_DEV_Q) if N_DEV_Q else DEV_Q_ALL
print(f"fuser fitting: {len(FITQ)} questions on {len(VAL_DBS)} databases the decider never saw (pilot used {len(VAL150)}) | dev: {len(DEVQ)}")

# ---------------------------------------------------------------- 4  hybrid search, embeddings for tables and columns
say("step 3/7: hybrid search and embeddings")
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
words = lambda s: re.findall(r"[a-z0-9]+", s.lower().replace("_", " "))
bm = BM25Okapi([words(c) for c in CARD])
emb = SentenceTransformer(EMBEDDER, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
TE = emb.encode(CARD, normalize_embeddings=True, batch_size=32)
CE = emb.encode([c[1] for c in COLTXT], normalize_embeddings=True, batch_size=64)
COL_T = np.array([c[0] for c in COLTXT])
COLS_OF = [np.where(COL_T == j)[0] for j in range(len(TABLES))]
rankm = lambda X: np.argsort(np.argsort(-X, 1), 1)


def search(qs):
    QE = emb.encode([qtext(q) for q in qs], prompt_name="query", normalize_embeddings=True, batch_size=64)
    DE = QE @ TE.T
    BM = np.stack([bm.get_scores(words(qtext(q))) for q in qs])
    return dict(QE=QE, DE=DE, BM=BM, S=1 / (60 + rankm(DE)) + 1 / (60 + rankm(BM)))


SR = {"fit": search(FITQ), "val150": search(VAL150), "dev": search(DEVQ)}
for k in SR: SR[k]["QC"] = SR[k]["QE"] @ CE.T                            # question x column similarities
del emb; gc.collect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------- 5  decider on the top-100 candidates
say("step 4/7: decider scores the 100 candidates of each question")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(str(DEC_DIR), num_labels=1, trust_remote_code=True).cuda().eval()
tok = AutoTokenizer.from_pretrained(str(DEC_DIR), trust_remote_code=True)


@torch.no_grad()
def decide(name, qs):
    f = OUT / f"dec_{name}_{len(qs)}.npz"
    CAND = np.argsort(-SR[name]["S"], 1)[:, :K_CAND]
    if f.exists(): return CAND, np.load(f)["D"]
    D = np.zeros(CAND.shape, dtype=np.float32)
    for i, q in enumerate(qs):
        x = tok([qtext(q)] * K_CAND, [CARD[j] for j in CAND[i]], truncation=True, max_length=384, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            D[i] = model(**x).logits[:, 0].float().cpu().numpy()
        if i % 300 == 0: say(f"  {name}: {i}/{len(qs)}")
    np.savez(f, D=D); return CAND, D


for k, qs in (("fit", FITQ), ("val150", VAL150), ("dev", DEVQ)):
    SR[k]["CAND"], SR[k]["D"] = decide(k, qs)
del model; gc.collect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------- 6  features, by family
say("step 5/7: features")
from rapidfuzz.process import cdist
from rapidfuzz.distance import Indel
FAM = {}                                                                 # family -> list of feature names (filled on first row)


def features(name, qs):
    R = SR[name]; rows, labels, groups = [], [], []
    for i, q in enumerate(qs):
        cand, D = R["CAND"][i], R["D"][i]
        S, DE, BM, QC = R["S"][i, cand], R["DE"][i, cand], R["BM"][i, cand], R["QC"][i]
        BMn = (BM - BM.min()) / (BM.max() - BM.min() + 1e-9)
        rS, rD = np.argsort(np.argsort(-S)), np.argsort(np.argsort(-D))
        dbs = [TABLES[j][0] for j in cand]
        dmax, dn, dmass, smax = {}, {}, {}, {}
        sm = np.exp(D - D.max()); sm /= sm.sum(); Dsort = np.sort(D)[::-1]
        for k, d in enumerate(dbs):
            dmax[d] = max(dmax.get(d, -1e9), D[k]); dn[d] = dn.get(d, 0) + 1; dmass[d] = dmass.get(d, 0) + sm[k]
            smax[d] = max(smax.get(d, 0), S[k])
        dbrank = {d: r for r, d in enumerate(sorted(dmax, key=lambda d: -dmax[d]))}
        rin = {}
        for k in np.argsort(-D):
            rin.setdefault(dbs[k], []).append(k)
        rank_in_db = {k: r for d in rin for r, k in enumerate(rin[d])}
        top1D, top5D, top5S = cand[np.argmax(D)], set(cand[np.argsort(-D)[:5]]), set(cand[np.argsort(-S)[:5]])
        qt = set(toks(q["question"])); ev = q.get("evidence", "") or ""; evl = ev.lower(); ql = q["question"].lower()
        qtall = qt | set(toks(ev))
        # fuzzy match of every question word against the catalog vocabulary (character-level)
        qw = [w for w in qtall if len(w) >= 4]
        fz = cdist(qw, VOCAB, scorer=Indel.normalized_similarity, workers=-1) if qw else np.zeros((0, len(VOCAB)))
        best_tok = fz.max(0) if len(qw) else np.zeros(len(VOCAB))
        Dz, DEz = (D - D.mean()) / (D.std() + 1e-6), (DE - DE.mean()) / (DE.std() + 1e-6)
        g = set(gold_ids(q))
        for k, j in enumerate(cand):
            d = dbs[k]
            cs = np.sort(QC[COLS_OF[j]])[::-1] if NCOL[j] else np.zeros(1)
            ct = CTOK[j]; ids = [VID[w] for w in ct]
            evcols = sum(1 for c, nat in COLS[j] if re.search(r"(?<![\w])" + re.escape(c.lower()) + r"(?![\w])", evl))
            qnat = sum(1 for c, nat in COLS[j] if len(nat) > 3 and nat.lower() in ql)
            f = {
                # base: the pilot's score features
                "base:hyb": S[k], "base:hyb_rank": rS[k], "base:dense": DE[k], "base:bm25": BMn[k], "base:dec": D[k], "base:dec_rank": rD[k],
                "base:db_n": dn[d], "base:db_maxdec": dmax[d], "base:db_rank": dbrank[d], "base:dec_minus_dbmax": D[k] - dmax[d],
                "base:db_size": len(TBL_OF[d]), "base:hyb_dbmax": smax[d],
                # column level: match the question to the best columns, not the averaged table
                "col:max": cs[0], "col:top2": cs[:2].mean(), "col:top3": cs[:3].mean(), "col:max_minus_table": cs[0] - DE[k],
                "col:n_over_0.5": float((cs > 0.5).sum()), "col:ncols": NCOL[j],
                # lexical: exact, IDF-weighted and fuzzy token overlap
                "lex:tname_overlap": len(qtall & TTOK[j]) / max(1, len(TTOK[j])), "lex:col_overlap": len(qtall & ct) / max(1, len(ct)),
                "lex:idf_overlap": sum(IDF[w] for w in qtall & ct), "lex:n_overlap": len(qtall & ct),
                "lex:fuzzy_max": max((best_tok[x] for x in ids), default=0), "lex:fuzzy_n_over_0.8": sum(best_tok[x] > 0.8 for x in ids),
                "lex:col_fullname_in_q": qnat,
                # evidence: BIRD's hint text often names tables and columns
                "ev:table_named": float(bool(re.search(r"(?<![\w])" + re.escape(TABLES[j][1].lower()) + r"(?![\w])", evl))),
                "ev:n_cols_named": evcols, "ev:has_evidence": float(bool(ev)),
                # schema graph: join structure around the top hits
                "graph:degree": len(ADJ[j]), "graph:nbr_of_top1": float(top1D in ADJ[j]), "graph:n_nbr_top5_dec": len(ADJ[j] & top5D),
                "graph:n_nbr_top5_hyb": len(ADJ[j] & top5S), "graph:bridge": float(len(ADJ[j] & top5D) >= 2),
                # relative: position among this question's candidates
                "rel:dec_z": Dz[k], "rel:dense_z": DEz[k], "rel:rank_in_db": rank_in_db[k], "rel:db_mass": dmass[d],
                "rel:dec_gap_next": D[k] - Dsort[min(rD[k] + 1, len(D) - 1)],
                # embedding interaction: question x table (compressed below)
            }
            if not FAM:
                for n in f: FAM.setdefault(n.split(":")[0], []).append(n)
            rows.append([f[n] for fam in FAM.values() for n in fam]); labels.append(int(j in g)); groups.append(i)
    X = np.array(rows, dtype=np.float32)
    QT = R["QE"][np.array(groups)] * TE[R["CAND"].reshape(-1)]            # elementwise question x table embedding product
    return X, np.array(labels), np.array(groups), QT


t0 = time.time()
DATA = {k: features(k, qs) for k, qs in (("fit", FITQ), ("val150", VAL150), ("dev", DEVQ))}
from sklearn.decomposition import PCA
pca = PCA(16, random_state=0).fit(DATA["fit"][3][::7])
names = [n for fam in FAM.values() for n in fam] + [f"emb:pc{p}" for p in range(16)]
FAM["emb"] = [f"emb:pc{p}" for p in range(16)]
DATA = {k: (np.hstack([X, pca.transform(QT)]), y, g) for k, (X, y, g, QT) in DATA.items()}
say(f"  {len(names)} features in {len(FAM)} families ({(time.time() - t0) / 60:.1f} min)")
col = {n: i for i, n in enumerate(names)}

# ---------------------------------------------------------------- 7  fusers and evaluation
say("step 6/7: fitting fusers")
from lightgbm import LGBMRanker
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import warnings; warnings.filterwarnings("ignore")
Xd, yd, gd = DATA["dev"]
NG = np.bincount(gd)


def evaluate(score):
    """Per question: recall and complete at k, ranking the 100 candidates by score."""
    out = {}
    for k in (5, 10, 15, 30):
        r, c = [], []
        for i in range(len(DEVQ)):
            m = gd == i; y = yd[m]; ng = len(gold_ids(DEVQ[i]))
            hit = y[np.argsort(-score[m])[:k]].sum()
            r.append(hit / ng); c.append(float(hit == ng))
        out[k] = (np.array(r), np.array(c))
    return out


def ranker(features, Xf, yf, gf):
    idx = [col[n] for n in features]
    m = LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    m.fit(Xf[:, idx], yf, group=np.bincount(gf))
    return m.predict(Xd[:, idx]), m


Xf, yf, gf = DATA["fit"]; X150, y150, g150 = DATA["val150"]
BASE = FAM["base"]; ALL = names
RES = {}
RES["hybrid search alone"] = evaluate(Xd[:, col["base:hyb"]])
RES["decider alone"] = evaluate(Xd[:, col["base:dec"]])
ib = [col[n] for n in BASE]
stack = StackingClassifier(estimators=[
    ("rf", RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=0)),
    ("lgb", LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1)),
    ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)))],
    final_estimator=LogisticRegression(max_iter=3000), cv=5).fit(X150[:, ib], y150)
RES[f"A  pilot fuser: {len(BASE)} score features, stacked classifier, {len(VAL150)} questions"] = evaluate(stack.predict_proba(Xd[:, ib])[:, 1])
RES[f"B  same {len(BASE)} features, lambdarank, {len(FITQ)} questions"] = evaluate(ranker(BASE, Xf, yf, gf)[0])
sc, mC = ranker(ALL, Xf, yf, gf)
RES[f"C  all {len(ALL)} engineered features, lambdarank, {len(FITQ)} questions"] = evaluate(sc)
for fam in FAM:
    if fam == "base": continue
    RES[f"   C without {fam} ({len(FAM[fam])})"] = evaluate(ranker([n for n in ALL if n not in FAM[fam]], Xf, yf, gf)[0])

# ---------------------------------------------------------------- report
say("step 7/7: report")
multi = np.array([len(gold_ids(q)) >= 2 for q in DEVQ])
L = [f"Fuser directional test | BIRD dev {len(DEVQ)} questions ({multi.sum()} need 2+ tables) | candidates: hybrid top {K_CAND} "
     f"| pooled catalog {len(TABLES)} tables | decider from pilot {PILOT_TAG}", "",
     f"{'fuser':66s} {'rec@15':>7s} {'cmp@5':>6s} {'cmp@10':>7s} {'cmp@15':>7s} {'cmp@30':>7s} {'cmp@15 2+tbl':>12s}"]
for k, v in RES.items():
    L.append(f"{k:66s} {100*v[15][0].mean():7.1f} {100*v[5][1].mean():6.1f} {100*v[10][1].mean():7.1f} {100*v[15][1].mean():7.1f} "
             f"{100*v[30][1].mean():7.1f} {100*v[15][1][multi].mean():12.1f}")
a = list(RES)[2]; c = list(RES)[4]
rs = np.random.default_rng(0); diff = RES[c][15][1] - RES[a][15][1]; diff10 = RES[c][10][1] - RES[a][10][1]
bs = [diff[rs.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
bs10 = [diff10[rs.integers(0, len(diff10), len(diff10))].mean() for _ in range(2000)]
L += ["", f"C minus A, complete@15: {100*diff.mean():+.1f} points (95% CI {100*np.percentile(bs, 2.5):+.1f} to {100*np.percentile(bs, 97.5):+.1f})",
      f"C minus A, complete@10: {100*diff10.mean():+.1f} points (95% CI {100*np.percentile(bs10, 2.5):+.1f} to {100*np.percentile(bs10, 97.5):+.1f})",
      "", "top 15 features of C by gain:"]
gain = mC.booster_.feature_importance("gain"); order = np.argsort(-gain)[:15]
L += [f"  {names[i]:28s} {100 * gain[i] / gain.sum():5.1f}%" for i in order]
L += ["", "complete@k = share of questions with ALL gold tables in the top k. Values (cell contents) are not used yet: train databases have no sqlite here."]
(OUT / "FEATURES.txt").write_text("\n".join(L) + "\n")
print("\n" + "\n".join(L))
