# Google Open Knowledge Format (OKF) — Zero to Mastery

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![OKF v0.1](https://img.shields.io/badge/OKF-v0.1-green)
![Local Ollama](https://img.shields.io/badge/models-local%20Ollama%20≤4B-informational)

Two fully-executed Jupyter notebooks, no cloud LLM APIs, no fabricated outputs.

1. **`google_okf_zero_to_mastery.ipynb`** — implements Google's [Open Knowledge Format v0.1](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md) from scratch: turns an undocumented, multi-table production dataset into an agent-consumable knowledge bundle, plus a local, grounded question-answering agent over it.
2. **`agentic_rag_chromadb.ipynb`** — the traditional alternative to OKF on the same dataset: no knowledge bundle at all, just flat text chunks in [ChromaDB](https://www.trychroma.com/), consumed by an **agentic** tool-calling loop instead of a fixed retrieval step.

Every number in this README is copied from an executed cell output, not estimated.

## Table of contents

- [Why this exists](#why-this-exists)
- [Notebook 1: OKF from scratch](#notebook-1-okf-from-scratch)
- [Notebook 2: Agentic RAG and ChromaDB](#notebook-2-agentic-rag-and-chromadb)
- [OKF vs RAG and VectorDB](#okf-vs-rag-and-vectordb)
- [Repository structure](#repository-structure)
- [Running it](#running-it)
- [License](#license)

## Why this exists

- **Dataset**: [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — real, anonymized, ~100k orders, 9 relational tables, CC BY-NC-SA 4.0 — downloaded live from Kaggle by both notebooks, never committed to this repo.
- **Models**: local Ollama models only, both ≤4B params, identical across both notebooks — `qwen3.5:4b` for generation (thinking disabled) and `nomic-embed-text` for embeddings. No API keys beyond Kaggle credentials.
- **The question both notebooks answer differently**: given a genuinely undocumented dataset, how should an AI agent's knowledge about it be represented and retrieved? Notebook 1 answers "as a structured, spec-conformant knowledge bundle." Notebook 2 answers "as flat chunks in a vector database, retrieved by an autonomous agent" — the answer most real-world RAG systems reach for by default.

## Notebook 1: OKF from scratch

**`google_okf_zero_to_mastery.ipynb`**

| # | Section |
|---|---|
| 1 | Theory — the full OKF v0.1 spec, explained and worked through with real code |
| 2 | The real problem — Olist's 9 undocumented CSVs, downloaded and profiled |
| 3 | An OKF core library written from scratch against the spec |
| 4 | Deterministic foreign-key discovery + real computed business metrics (pandas, not the LLM) |
| 5 | A local LLM enrichment agent — facts come from code, the model only writes prose, every generated claim is checked against ground truth |
| 6 | Conformance validation against SPEC.md §9 |
| 7 | Interactive graph visualization of the bundle (self-contained HTML) |
| 8 | A local RAG discovery agent (FAISS + local-LLM rerank) answering real questions with citations |
| 9 | Honest evaluation — retrieval accuracy, rerank-vs-plain-retrieval comparison, answer groundedness |
| 10 | Mastery recap vs. Google's reference implementation, plus stated limitations |

**Real results from the last full run:**

- Generated bundle: **16 conformant OKF concepts** (1 dataset, 9 tables, 6 metrics) — 0 conformance errors, 0 warnings.
- Retrieval accuracy on 8 hand-labeled questions: **Hit@1 88%, Hit@3/Hit@5 100%**.
- Answer groundedness on 3 numeric questions: **3/3** — generated answers contain the true computed number.
- Two real failure modes were found and fixed *during development*, documented rather than hidden:
  - A 4B model asked to freely rerank retrieval candidates dropped the single most relevant document for one question — fixed with a guaranteed floor of raw retrieval hits (effect measured in Part 10.1).
  - An underspecified prompt caused the model to confidently invent a date range for the dataset ("2019–2023" vs. the real 2016-09-04 to 2018-10-17) — fixed by grounding the prompt with the real range and verifying no other year appears in the output.

## Notebook 2: Agentic RAG and ChromaDB

**`agentic_rag_chromadb.ipynb`** — no OKF at all, the traditional alternative.

Fully standalone — same dataset, same models, same grounding discipline and hallucination guards as notebook 1, but the output is **flat `{id, text, metadata}` documents**, not OKF concepts: no frontmatter, no bundle hierarchy, no parseable link graph. Ingested into a ChromaDB collection with a custom Ollama embedding function, then consumed two ways:

- **Simple RAG** — one retrieval call, one generation call (a fair control, architecturally identical in shape to notebook 1's baseline).
- **Agentic RAG** — native Ollama tool-calling (cited to [Yao et al. 2022, ReAct](https://arxiv.org/abs/2210.03629) and the [Agentic RAG survey](https://arxiv.org/abs/2501.09136)), where the model itself decides whether one search is enough or whether to search again, bounded at 4 iterations.

Evaluated on the same 14-question hand-labeled set as the retired advanced-RAG benchmark (direct / paraphrase / multi-hop / vague / unanswerable-distractor).

**Real head-to-head results from the last full run** (12 answerable, 2 distractors, simple vs. agentic — a controlled, same-run, same-corpus comparison):

| Metric | Simple RAG | Agentic RAG |
|---|---|---|
| Retrieval hit rate | 91.7% | 91.7% |
| Answer-citation hit rate | **83.3%** | 75.0% |
| Refusal correctness (distractors) | 100% | 100% |
| Avg searches issued / question | 1 (fixed) | 1.17 |
| Avg LLM+embed calls / question | 2.0 | 3.29 (**1.6×**) |

**The real finding isn't the scorecard, it's the 1.17.** The agent almost never chose to use its own autonomy — most questions got exactly one search, behaviorally identical to the simple pipeline. The exception was the multi-hop category, where it averaged **1.67 searches/question** and *won* clearly (100% vs. 66.7% citation-hit) — the one place genuinely needing more than one lookup is exactly where the agent chose to make more than one. Elsewhere (paraphrase: 66.7% vs. 100%) the extra autonomy bought nothing and slightly hurt. 0/14 questions hit the iteration cap. This mirrors the pattern from this project's now-retired advanced-RAG-vs-baseline benchmark almost exactly: added machinery helps precisely where it's structurally motivated, and adds noise or cost everywhere else.

**Not a controlled comparison to notebook 1** — different corpus representation, different architecture end-to-end, and notebook 1 was scored on an easier 8-question set in a separate execution. See notebook 2, Part 8, for a hands-on architectural comparison instead of an apples-to-oranges number.

## OKF vs RAG and VectorDB

How they actually relate.

These solve different problems and are not competitors — this project built both, on the same data, to make the tradeoff concrete rather than assert it.

| | **OKF (notebook 1)** | **Flat RAG + VectorDB (notebook 2)** |
|---|---|---|
| **What it is** | A knowledge *representation* spec — markdown + YAML frontmatter, bundles, concepts, cross-links | An *inference-time retrieval technique* over unstructured chunks in a vector database |
| **Layer** | Authoring / storage | Query-time consumption |
| **Storage** | Plain files, `git`-diffable, human-browsable on GitHub with zero tooling | A running (even if embedded/in-process) vector database process |
| **Structure** | Explicit: typed frontmatter, `# Schema`/`# Joins`/`# Citations` sections, a real link graph | None inherent — whatever metadata fields the pipeline author remembered to attach |
| **Consumption without retrieval** | Possible — a small bundle fits wholesale in context, browsable via `index.md` | Not really — there's no directory to browse; the vector index *is* the access path |
| **Conformance / validation** | A real, checkable spec (SPEC.md §9) — this project's validator actually runs and asserts | No equivalent — "is this corpus well-formed" isn't a question a vector DB asks |
| **Consumption pattern used here** | Dense retrieval + guaranteed-floor listwise rerank | Simple single-shot retrieval, or an agentic tool-calling loop that decides for itself |
| **Setup cost** | A YAML/markdown parser, nothing else | A vector DB dependency (ChromaDB — embedded SQLite + ONNX runtime) |
| **Where it shines** | Long-lived, versioned, human-*and*-agent-curated knowledge that outlives any one retrieval pipeline | Fast to stand up, works over content nobody structured on purpose (PDFs, wikis, scraped pages) |
| **Failure modes measured here** | Hallucinated facts from an underspecified generation prompt (dates, metric values) — fixed via explicit grounding + retry-then-fallback verification | Listwise-rerank unreliability at small model scale (notebook 1); an agent that mostly doesn't use its own autonomy unless the question structurally needs it (notebook 2) |

**The one-sentence version**: OKF is what you retrieve *from*; RAG is *how* you retrieve it. A corpus this small (16 items either way) doesn't strictly need either structure or agentic retrieval — a bundle could be loaded wholesale into context, and a single search already finds the right answer 92% of the time. Both notebooks were built specifically to *measure* that rather than assume more machinery is automatically better.

## Repository structure

```
google-okf-tutorial/
├── google_okf_zero_to_mastery.ipynb   # Notebook 1 (OKF from scratch)
├── agentic_rag_chromadb.ipynb         # Notebook 2 (agentic RAG + ChromaDB, no OKF)
├── pyproject.toml / uv.lock           # dependency manifest (uv)
├── LICENSE                            # MIT (code only — see License section)
├── README.md
└── bundle/                            # generated at runtime by notebook 1, git-ignored
```

## Running it

```bash
uv sync
ollama pull qwen3.5:4b
ollama pull nomic-embed-text
uv run jupyter lab google_okf_zero_to_mastery.ipynb   # notebook 1
uv run jupyter lab agentic_rag_chromadb.ipynb         # notebook 2 (standalone)
```

Kaggle credentials must be available (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`) for the dataset download cell in each notebook.

Notebook 1 regenerates `bundle/` (the produced OKF knowledge base, including the interactive `viz.html` graph) — git-ignored since it's a reproducible build artifact, not source. Notebook 2 uses an ephemeral (in-memory) ChromaDB collection that exists only for the notebook's runtime; nothing persists to disk.

## License

The **code** in this repository (notebooks, configuration) is licensed under the [MIT License](LICENSE).

The **Olist dataset** is CC BY-NC-SA 4.0 (verified via the Kaggle API's `dataset-metadata.json`, not assumed) — non-commercial, share-alike, attribution required. It is downloaded at runtime and never redistributed in this repository. The generated `bundle/` directory is a derived documentation artifact for **educational use only** and is git-ignored, so it is never distributed as part of this repo either.
