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

No data is included. `main.py` downloads BIRD, Spider and Fiben from their public sources. BEAVER must be obtained from
its authors' public release and placed in `BEAVER_DIR`. Please follow each benchmark's licence.

## Results

The paper reports every result. Per-question outputs are written to `OUT_DIR` when the cell runs
(`FINAL.txt` and the review-analysis report).

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

NIFT-SQL (arXiv:2611.12731) studies the complementary case: models trained on the target customer's own questions.

## Licence

No licence file has been added yet. Until one is, the default copyright rules apply: you may read the code, but reuse
needs the authors' permission.
