# Google Open Knowledge Format (OKF) — Zero to Mastery

A single, fully-executed Jupyter notebook that implements Google's [Open Knowledge Format v0.1](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md) from scratch and solves a real problem with it: turning an undocumented, multi-table production dataset into an agent-consumable knowledge bundle, plus a local, grounded question-answering agent over it.

Everything runs locally and for real — no cloud APIs beyond the one-time Kaggle download.

- **Dataset**: [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (real, anonymized, ~100k orders, 9 relational tables, CC BY-NC-SA 4.0), downloaded live from Kaggle by the notebook.
- **Models**: local Ollama models, both ≤ 4B params — `qwen3.5:4b` (generation, thinking disabled) and `nomic-embed-text` (embeddings). No API keys beyond Kaggle credentials.
- **OKF implementation**: an independent implementation of the v0.1 spec — document model, frontmatter render/parse, bundle I/O, index/log generation, cross-linking, and a real conformance validator.

## What's inside

1. Theory — the full OKF v0.1 spec, explained and worked through with real code.
2. The real problem — Olist's 9 undocumented CSVs, downloaded and profiled.
3. An OKF core library written from scratch against the spec.
4. Deterministic foreign-key discovery + real computed business metrics (pandas, not the LLM).
5. A local LLM enrichment agent that drafts OKF concept documents — facts come from code, the model only writes prose, and every generated claim is checked against ground truth.
6. Conformance validation against SPEC.md §9.
7. An interactive graph visualization of the bundle (self-contained HTML).
8. A local RAG discovery agent (FAISS + local-LLM rerank) that answers real questions with citations.
9. Honest evaluation: retrieval accuracy, a measured comparison of rerank vs. plain retrieval, and answer-groundedness checks.
10. A mastery recap comparing this build to Google's reference implementation, plus stated limitations.

## Real results from the last full run

- Generated bundle: **16 conformant OKF concepts** (1 dataset, 9 tables, 6 metrics), 0 conformance errors, 0 warnings.
- Retrieval accuracy on 8 hand-labeled questions: **Hit@1 88%, Hit@3/Hit@5 100%**.
- Answer groundedness on 3 numeric questions: **3/3** — generated answers contain the true computed number.
- Two real failure modes were found and fixed *during development*, not hidden:
  - A 4B model asked to freely rerank retrieval candidates dropped the single most relevant document for one question — fixed with a guaranteed floor of raw retrieval hits, and the effect is measured directly in the notebook (Part 10.1).
  - An underspecified prompt caused the model to confidently invent a date range for the dataset ("2019–2023" vs. the real 2016-09-04 to 2018-10-17) — fixed by grounding the prompt with the real range and verifying no other year appears in the output.

## Running it

```bash
uv sync
ollama pull qwen3.5:4b
ollama pull nomic-embed-text
uv run jupyter lab google_okf_zero_to_mastery.ipynb
```

Kaggle credentials must be available (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`) for the dataset download cell.

Running the notebook regenerates `bundle/` (the produced OKF knowledge base, including the interactive `viz.html` graph) — it's git-ignored since it's a reproducible build artifact, not source.

## License note

The Olist dataset is CC BY-NC-SA 4.0 (verified via the Kaggle API, not assumed) — non-commercial, share-alike, attribution required. This repository is for educational purposes.
