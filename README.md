# FERN

FERN = Feature-Engineered Ranking with No generative LLM.

Code for the paper:

> Arjun Rao and Nishant Gopinath. **FERN: Cross-Benchmark Table Retrieval for Text-to-SQL with Frozen Retrievers and
> Learned Ranking.** arXiv:2611.14477, 2026. https://arxiv.org/abs/2611.14477

FERN finds the tables a text-to-SQL question needs in a catalog that pools the tables of many databases, without
calling a large language model. It has three steps:

1. **Candidates.** The top 100 tables by reciprocal rank fusion of three frozen retrievers: BM25,
   Qwen3-Embedding-0.6B and BGE-large-en-v1.5. None is fine-tuned.
2. **Features.** 120 numbers per (question, candidate table) in 12 families: retrieval scores, agreement between
   retrievers, database-level support, lexical matches, question-word coverage, BIRD evidence, cell values, column
   types, table metadata, join graph, question context, and a compressed embedding interaction.
3. **Ranker.** LightGBM lambdarank, fitted once on BIRD training questions. Nothing is fitted on Spider, Fiben or
   BEAVER.

Three rankers are fitted on BIRD train only and differ in which inputs they use: with BIRD evidence (120 features),
question only (116), and schema only, without cell values, column types or row counts (102).

## What is in this repository

```
main.py            the frozen code that produced the paper's results, top to bottom, as one Colab cell
development/       the earlier selection and development cells (pilot, features, out-of-fold, direction)
v2/                loader for the public BEAVER release and a script that writes its declared-key file
requirements.txt   the packages main.py installs
```

`main.py` is published unchanged, so it is the code behind the paper's numbers. It was written as a single cell for
Google Colab (one A100 40GB GPU): it installs its packages, mounts Google Drive for caching, downloads the data, and
runs every stage in order. The stages are numbered in the file:

1. install
2. data: BIRD, Spider, Fiben, BEAVER
3. catalogs and gold tables (parsed from the gold SQL with sqlglot)
4. frozen scorers
5. features
6. BIRD train: features and the three rankers
7. evaluation: every test set scored once
8. checks: leakage audit, feature dump, latency

The file also contains two blocks that run from saved outputs: the review analyses reported in the paper
(`RUN_ANALYSIS = True`) and an execution-accuracy appendix that is **not part of the paper** and is off by default
(`RUN_EX = False`).

## Run it

Open a Colab notebook with a GPU, paste the whole of `main.py` into one cell, set the four paths at the top, and run.
The first run takes about 1.5 hours; every stage is cached in `OUT_DIR`, so the cell can be re-run after a runtime
restart.

| Setting | Meaning |
|---|---|
| `OUT_DIR` | where results and caches are written |
| `OOF_DIR` | optional: an existing BIRD cell-value index is reused if present |
| `BEAVER_DIR` | a folder holding the public BEAVER release |
| `KEYS_FILE` | optional: BEAVER declared foreign keys, written by `python v2/schema_keys.py --data <BEAVER_DIR> --out <DIR>` |

## Data

No data is included; follow each benchmark's licence. `main.py` downloads:

| Data | Source used by `main.py` |
|---|---|
| BIRD dev and train | the official archives at `bird-bench.oss-cn-beijing.aliyuncs.com` |
| Spider | the Hugging Face dataset `HAL-9001/spider-databases`, a community copy of Spider that includes its databases |
| Fiben | `git clone --depth 1` of `github.com/IBM/fiben-benchmark` |

BEAVER is gated, so it is downloaded once by hand: accept the terms of the Hugging Face datasets
`beaverbench/beaver-query` and `beaverbench/beaver-table`, then, with your token in `HF_TOKEN`,

```
python -c "from huggingface_hub import snapshot_download as s; import os
for r in ('beaverbench/beaver-query', 'beaverbench/beaver-table'):
    s(r, repo_type='dataset', token=os.environ['HF_TOKEN'], local_dir='<BEAVER_DIR>/' + r.split('/')[1])"
python v2/schema_keys.py --data <BEAVER_DIR> --out <DIR>      # optional: declared foreign keys -> <DIR>/schema_keys.json
```

`main.py` finds the parquet files anywhere under `BEAVER_DIR`.

## Reproducibility

**Exact code.** `main.py` is the authors' final cell, published unchanged. It is written for Google
Colab (it mounts Google Drive and expects a GPU), so it runs there as published.

**What is fixed.** The method has no tuning on any test set, the rankers are fitted once on BIRD train, and the random
parts are seeded (LightGBM, PCA, the bootstrap and the question samples all use fixed seeds).

**What can move.** Three things are not pinned, so a rerun can differ slightly from the paper:

- *Package versions.* The cell installs the current versions of its packages each time it runs, and the versions used
  for the paper (Google Colab, November 2026) were not recorded. A newer sentence-transformers, LightGBM or sqlglot
  can shift scores, rankings or parsed gold tables.
- *Data snapshots.* Fiben is cloned at its latest commit and Spider comes from a community mirror; neither is pinned to
  a revision.
- *Numerics.* Embeddings computed on a GPU, and LightGBM run with many threads, are not bit-for-bit repeatable, and
  ties in a ranking can then break differently.

A rerun should therefore reproduce the paper's conclusions and land within its confidence intervals, not every digit.

## Results

When the cell runs, it writes its reports to `OUT_DIR`: `FINAL.txt` (every test set, scored once) and `ANALYSIS.txt`
(the review analyses reported in the paper).

## Citation

```
@article{rao2026fern,
  title   = {FERN: Cross-Benchmark Table Retrieval for Text-to-SQL with Frozen Retrievers and Learned Ranking},
  author  = {Rao, Arjun and Gopinath, Nishant},
  journal = {arXiv preprint arXiv:2611.14477},
  year    = {2026}
}
```

## Related work by the same authors

NIFT-SQL (arXiv:2611.12731; code: https://github.com/arsr/nift-sql-publish) studies the complementary case: models
trained on the target customer's own questions. FERN's two BEAVER comparison rows from the NIFT-SQL holdout come from
that repository (`experiments/confirm_beaver_v3.py`, `results/paper/CONFIRM.txt`).

## Licence

No licence file has been added yet. Until one is, the default copyright rules apply: you may read the code, but reuse
needs the authors' permission.
