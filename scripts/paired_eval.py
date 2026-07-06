#!/usr/bin/env python3
"""
Paired, side-by-side comparison across:
- OKF bundle + FAISS retrieval + hybrid rerank (notebook 1-style)
- Flat corpus + ChromaDB simple RAG (notebook 2-style)
- Flat corpus + ChromaDB agentic tool-calling RAG (notebook 2-style)

This script is intentionally notebook-derived so results match the repo's narrative.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import chromadb
import faiss
import kagglehub
import numpy as np
import ollama
import pandas as pd
import yaml
from chromadb import EmbeddingFunction
from loguru import logger


GEN_MODEL = "qwen3.5:4b"
EMBED_MODEL = "nomic-embed-text"
KAGGLE_DATASET = "olistbr/brazilian-ecommerce"
KAGGLE_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
REFUSAL_RE = re.compile(
    r"do(?:es)? not (?:contain|include|provide)|not available|cannot determine|no information|"
    r"don'?t have|isn'?t available|unable to (?:determine|answer)|not (?:mentioned|specified|available)|"
    r"knowledge base (?:entries )?do(?:es)? not|no results found",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvalItem:
    q: str
    expected: set[str]
    answerable: bool
    category: str


def looks_like_refusal(answer: str) -> bool:
    return bool(REFUSAL_RE.search(answer))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class CallCounter:
    def __init__(self) -> None:
        self.llm = 0
        self.embed = 0

    def snapshot(self) -> dict[str, int]:
        return {"llm": self.llm, "embed": self.embed}

    def delta(self, before: dict[str, int]) -> int:
        return (self.llm - before["llm"]) + (self.embed - before["embed"])


def model_available(name: str, local_models: set[str]) -> bool:
    return name in local_models or any(m.split(":")[0] == name.split(":")[0] for m in local_models)


def llm_text(counter: CallCounter, prompt: str, *, num_predict: int, temperature: float, retries: int = 2) -> str:
    counter.llm += 1
    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            resp = ollama.chat(
                model=GEN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                think=False,
                options={"num_predict": num_predict, "temperature": temperature},
            )
            content = resp["message"]["content"].strip()
            if content:
                return content
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(f"LLM call failed: {last_err}")


def llm_chat(counter: CallCounter, messages: list[dict[str, Any]], *, tools: Optional[list] = None, temperature: float):
    counter.llm += 1
    return ollama.chat(
        model=GEN_MODEL,
        messages=messages,
        tools=tools,
        think=False,
        options={"temperature": temperature},
    )


def embed_one(counter: CallCounter, text: str) -> np.ndarray:
    counter.embed += 1
    resp = ollama.embed(model=EMBED_MODEL, input=text)
    return np.array(resp["embeddings"][0], dtype="float32")


class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, counter: CallCounter, model: str = EMBED_MODEL):
        self._counter = counter
        self.model = model

    def __call__(self, input):
        self._counter.embed += len(input)
        return [ollama.embed(model=self.model, input=t)["embeddings"][0] for t in input]

    def name(self):
        return f"ollama-{self.model}"


def load_eval_set(path: Path) -> list[EvalItem]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = raw["items"]
    out: list[EvalItem] = []
    for it in items:
        out.append(
            EvalItem(
                q=str(it["q"]),
                expected=set(map(str, it.get("expected", []))),
                answerable=bool(it["answerable"]),
                category=str(it["category"]),
            )
        )
    return out


def profile_table(name: str, df: pd.DataFrame) -> dict:
    cols = []
    for col in df.columns:
        s = df[col]
        cols.append(
            {
                "name": col,
                "dtype": str(s.dtype),
                "null_pct": round(float(s.isna().mean()) * 100, 2),
                "n_unique": int(s.nunique()),
                "samples": [str(v) for v in s.dropna().unique()[:3]],
            }
        )
    return {"name": name, "n_rows": int(len(df)), "n_cols": int(len(df.columns)), "columns": cols}


def base_key(col: str) -> str:
    return "zip_code_prefix" if col.endswith("zip_code_prefix") else col


def detect_fk_candidates(tables: dict[str, pd.DataFrame], *, min_overlap: float = 0.9) -> pd.DataFrame:
    from collections import defaultdict

    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for tname, df in tables.items():
        for col in df.columns:
            groups[base_key(col)].append((tname, col))
    rows: list[dict[str, Any]] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        for i, (t1, c1) in enumerate(members):
            for j, (t2, c2) in enumerate(members):
                if i == j:
                    continue
                s1, s2 = tables[t1][c1].dropna(), tables[t2][c2].dropna()
                if s1.empty or s2.empty:
                    continue
                overlap = float(s1.isin(set(s2.unique())).mean())
                if overlap >= min_overlap:
                    rows.append(
                        {
                            "from_table": t1,
                            "from_column": c1,
                            "to_table": t2,
                            "to_column": c2,
                            "overlap": round(overlap, 4),
                        }
                    )
    return pd.DataFrame(rows)


def format_metric_value(info: dict) -> str:
    v = info["value"]
    return ", ".join(f"{k}: {v2}" for k, v2 in v.items()) if isinstance(v, dict) else str(v)


def number_variants(x: float) -> set[str]:
    return {f"{x}", f"{x:.0f}", f"{x:.1f}", f"{x:.2f}", str(round(x))}


def metric_is_grounded(text: str, info: dict) -> bool:
    clean = text.replace(",", "")
    v = info["value"]
    candidates = v.values() if isinstance(v, dict) else [v]
    return any(variant in clean for x in candidates for variant in number_variants(float(x)))


def build_flat_corpus(
    counter: CallCounter,
    tables: dict[str, pd.DataFrame],
    profiles: dict[str, dict],
    fk_candidates: pd.DataFrame,
    metrics_summary: dict[str, dict],
) -> list[dict[str, Any]]:
    orders_df = tables["orders"].copy()
    for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        orders_df[col] = pd.to_datetime(orders_df[col], errors="coerce")

    def make_table_doc(name: str) -> dict:
        profile = profiles[name]
        col_summary = "; ".join(f"{c['name']} ({c['dtype']}, {c['null_pct']}% null)" for c in profile["columns"])
        rels = fk_candidates[(fk_candidates["from_table"] == name) | (fk_candidates["to_table"] == name)]
        rel_bits = [
            f"{r.from_table}.{r.from_column} relates to {r.to_table}.{r.to_column} ({r.overlap*100:.0f}% overlap)"
            for r in rels.itertuples()
        ]
        prompt = (
            "Document a database table. Use ONLY the facts given; never invent numbers or business meaning.\n\n"
            f"Table: {name}\nRows: {profile['n_rows']:,}\nColumns: {col_summary}\n"
            f"Relationships: {'; '.join(rel_bits) if rel_bits else 'none detected'}\n\n"
            "Write two short paragraphs separated by a blank line: (1) what one row represents, "
            "(2) a note on relationships/data quality. Output ONLY the prose."
        )
        raw = llm_text(counter, prompt, num_predict=250, temperature=0.2)
        title = name.replace("_", " ").title()
        text = (
            f"{title} (table)\n\n{raw}\n\n"
            f"Columns: {col_summary}\n"
            f"Relationships: {'; '.join(rel_bits) if rel_bits else 'none detected'}\n\n"
            f"Source: Olist Brazilian E-Commerce Public Dataset (Kaggle), table `{name}`."
        )
        return {"id": f"tables/{name}", "text": text, "metadata": {"type": "Table", "title": title}}

    def make_metric_doc(key: str) -> dict:
        info = metrics_summary[key]
        value_str = format_metric_value(info)
        prompt = (
            f"Document a business metric. Metric: {key.replace('_',' ')}\nValue: {value_str} {info['unit']}\n"
            f"Computed over: {info['n']:,} records\nDefinition: {info['definition']}\n"
            f"Write 2-3 sentences explaining the metric; you MUST include the exact value ({value_str}) verbatim. "
            "Output ONLY the prose."
        )
        text = llm_text(counter, prompt, num_predict=200, temperature=0.2)
        if not metric_is_grounded(text, info):
            text = llm_text(counter, prompt + f"\n\nReminder: {value_str} MUST appear verbatim.", num_predict=200, temperature=0.2)
        if not metric_is_grounded(text, info):
            text = f"The {key.replace('_', ' ')} is {value_str}, computed as {info['definition']} over {info['n']:,} records."
        title = key.replace("_", " ").title()
        full_text = (
            f"{title} (metric)\n\n{text}\n\n"
            f"Definition: {info['definition']}\nValue: {value_str} {info['unit']}\n"
            f"Computed over: {info['n']:,} records\nSource table(s): {', '.join(info['source_tables'])}."
        )
        return {"id": f"references/metrics/{key}", "text": full_text, "metadata": {"type": "Metric", "title": title}}

    def make_dataset_doc() -> dict:
        date_min, date_max = orders_df["order_purchase_timestamp"].min(), orders_df["order_purchase_timestamp"].max()
        valid_years = {date_min.year, date_max.year}
        table_list = "\n".join(f"- {n}: {p['n_rows']:,} rows" for n, p in profiles.items())
        prompt = (
            f"Document a dataset. Order date range: {date_min:%Y-%m-%d} to {date_max:%Y-%m-%d}\nTables:\n{table_list}\n\n"
            "Write 2-3 sentences on what this dataset represents. Use ONLY the facts above — do not state any "
            "date range other than the one given. Output ONLY the prose."
        )
        text = llm_text(counter, prompt, num_predict=200, temperature=0.2)

        def years_ok(t: str) -> bool:
            return {int(y) for y in re.findall(r"\b(20\d{2})\b", t)}.issubset(valid_years)

        if not years_ok(text):
            text = llm_text(
                counter,
                prompt + f"\n\nReminder: the only valid years are {sorted(valid_years)}.",
                num_predict=200,
                temperature=0.2,
            )
        if not years_ok(text):
            text = (
                "The Olist Brazilian E-Commerce Public Dataset captures orders placed between "
                f"{date_min:%B %Y} and {date_max:%B %Y} across {len(profiles)} tables."
            )
        full_text = (
            "Olist Brazilian E-Commerce (dataset)\n\n"
            f"{text}\n\nTables:\n{table_list}\n\n"
            f"Source: Olist Brazilian E-Commerce Public Dataset (Kaggle), {KAGGLE_URL}"
        )
        return {"id": "datasets/olist_ecommerce", "text": full_text, "metadata": {"type": "Dataset", "title": "Olist Brazilian E-Commerce"}}

    docs: list[dict[str, Any]] = [make_dataset_doc()]
    for name in profiles:
        docs.append(make_table_doc(name))
    for key in metrics_summary:
        docs.append(make_metric_doc(key))
    return docs


def parse_okf_concept(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(f"{path}: no parseable YAML frontmatter block")
    fm = yaml.safe_load(m.group(1)) or {}
    return fm, m.group(2)


def load_okf_concepts(bundle_root: Path) -> list[dict[str, Any]]:
    reserved = {"index.md", "log.md"}
    out = []
    for p in sorted(bundle_root.rglob("*.md")):
        if p.name in reserved:
            continue
        cid = str(p.relative_to(bundle_root).with_suffix("")).replace("\\", "/")
        fm, body = parse_okf_concept(p)
        out.append(
            {
                "id": cid,
                "type": fm.get("type"),
                "title": fm.get("title", cid),
                "description": fm.get("description", ""),
                "body": body,
            }
        )
    return out


def extract_bracket_citations(answer: str) -> list[str]:
    # Matches e.g. [tables/orders] and dedupes in order.
    cits = re.findall(r"\[([^\[\]]+?)\]", answer)
    seen = set()
    out = []
    for c in cits:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def build_okf_pipeline(counter: CallCounter, concepts: list[dict[str, Any]]):
    concept_by_id = {c["id"]: c for c in concepts}

    def embed_text_for(c: dict) -> str:
        return f"{c['title']}\n{c['description']}\n{c['body'][:1500]}"

    embeddings = np.stack([embed_one(counter, embed_text_for(c)) for c in concepts])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    vector_index = faiss.IndexFlatIP(embeddings.shape[1])
    vector_index.add(embeddings)
    id_by_row = [c["id"] for c in concepts]

    def retrieve(query: str, *, k: int) -> list[tuple[str, float]]:
        q = embed_one(counter, query)
        q /= np.linalg.norm(q)
        scores, idxs = vector_index.search(q.reshape(1, -1), k)
        return [(id_by_row[i], float(s)) for s, i in zip(scores[0], idxs[0]) if i != -1]

    def llm_rerank(query: str, candidate_ids: list[str], *, top_n: int) -> list[str]:
        listing = "\n".join(
            f"{i + 1}. {cid} — {concept_by_id[cid]['title']}: {concept_by_id[cid]['description']}"
            for i, cid in enumerate(candidate_ids)
        )
        prompt = (
            f'A user asked: "{query}"\n\nCandidate knowledge-base entries:\n{listing}\n\n'
            f"List the numbers of the {top_n} most relevant entries, most relevant first, as a comma-separated\n"
            'list of numbers only (e.g. "2, 5, 1"). Output ONLY the numbers.'
        )
        raw = llm_text(counter, prompt, num_predict=30, temperature=0.0)
        nums = [int(n) for n in re.findall(r"\d+", raw)]
        ranked = [candidate_ids[n - 1] for n in nums if 0 <= n - 1 < len(candidate_ids)]
        ranked = list(dict.fromkeys(ranked))
        ranked += [cid for cid in candidate_ids if cid not in ranked]
        return ranked[:top_n]

    def answer_question(query: str, *, k_retrieve: int, k_final: int, k_raw_guaranteed: int) -> dict[str, Any]:
        hits = retrieve(query, k=k_retrieve)
        candidate_ids = [cid for cid, _ in hits]
        llm_ranked = llm_rerank(query, candidate_ids, top_n=k_final)
        guaranteed = candidate_ids[:k_raw_guaranteed]
        final_context_ids = list(dict.fromkeys(guaranteed + llm_ranked))[:k_final]

        context = "\n\n---\n\n".join(
            f"[{cid}] {concept_by_id[cid]['title']}\n{concept_by_id[cid]['body']}" for cid in final_context_ids
        )
        prompt = (
            "Answer the user's question using ONLY the knowledge-base entries below. Cite the\n"
            "entry id(s) you used in square brackets, e.g. [tables/orders]. If the entries do not contain the\n"
            "answer, say so explicitly rather than guessing.\n\n"
            f"Knowledge base entries:\n{context}\n\nQuestion: {query}\n\nAnswer (2-4 sentences, with citations):"
        )
        answer = llm_text(counter, prompt, num_predict=300, temperature=0.1)
        return {
            "retrieved_ids": candidate_ids,
            "final_context_ids": final_context_ids,
            "answer": answer,
        }

    return answer_question


def build_flat_pipelines(counter: CallCounter, documents: list[dict[str, Any]]):
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(
        "olist_flat_corpus",
        embedding_function=OllamaEmbeddingFunction(counter, EMBED_MODEL),
    )
    collection.add(
        ids=[d["id"] for d in documents],
        documents=[d["text"] for d in documents],
        metadatas=[d["metadata"] for d in documents],
    )

    def simple_answer(query: str, *, top_k: int) -> dict[str, Any]:
        res = collection.query(query_texts=[query], n_results=top_k)
        ids, docs = res["ids"][0], res["documents"][0]
        context = "\n\n---\n\n".join(f"[{i}] {d}" for i, d in zip(ids, docs))
        prompt = (
            "Answer the user's question using ONLY the knowledge-base entries below. Cite entry "
            "id(s) in square brackets, e.g. [tables/orders]. If the entries do not contain the "
            "answer, say so explicitly rather than guessing.\n\n"
            f"Knowledge base entries:\n{context}\n\nQuestion: {query}\n\nAnswer (2-4 sentences, with citations):"
        )
        answer = llm_text(counter, prompt, num_predict=300, temperature=0.1)
        return {"retrieved_ids": list(ids), "answer": answer}

    SEARCH_TOOL = [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Semantic search over the knowledge base. Returns the top matching documents with their id and content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "the search query"},
                        "top_k": {"type": "integer", "description": "number of results to return (default 3)"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    AGENT_SYSTEM_PROMPT = (
        "You are a knowledge-base assistant. Use the search_knowledge_base tool to find relevant "
        "documents before answering. You may call it more than once if the first results are "
        "insufficient — for example, to look up a related table or metric mentioned in an earlier "
        "result. Once you have enough information, answer in 2-4 sentences, citing the document id(s) "
        "you used in square brackets, e.g. [tables/orders]. If nothing relevant turns up after "
        "searching, say so explicitly rather than guessing."
    )

    def _run_search_tool(args: dict) -> tuple[str, list[str]]:
        query = args.get("query", "")
        top_k = int(args.get("top_k", 3) or 3)
        res = collection.query(query_texts=[query], n_results=top_k)
        ids, docs = res["ids"][0], res["documents"][0]
        if not ids:
            return "No results found.", []
        return "\n\n".join(f"[{i}]\n{d[:600]}" for i, d in zip(ids, docs)), list(ids)

    def agentic_answer(query: str, *, max_iterations: int) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        search_log: list[dict[str, Any]] = []
        retrieved_ids: list[str] = []
        for step in range(max_iterations):
            resp = llm_chat(counter, messages, tools=SEARCH_TOOL, temperature=0.2)
            messages.append(resp["message"])
            tool_calls = resp["message"].get("tool_calls")
            if not tool_calls:
                return {
                    "answer": resp["message"]["content"],
                    "iterations": step + 1,
                    "searches": search_log,
                    "retrieved_ids": retrieved_ids,
                    "capped": False,
                }
            for tc in tool_calls:
                args = tc.function.arguments
                search_log.append({"query": args.get("query", query), "top_k": args.get("top_k", 3)})
                result, ids = _run_search_tool(args)
                for i in ids:
                    if i not in retrieved_ids:
                        retrieved_ids.append(i)
                messages.append({"role": "tool", "content": result, "tool_name": tc.function.name})

        messages.append({"role": "user", "content": "Answer now with what you already found, citing id(s)."})
        final = llm_chat(counter, messages, tools=None, temperature=0.2)
        return {
            "answer": final["message"]["content"],
            "iterations": max_iterations,
            "searches": search_log,
            "retrieved_ids": retrieved_ids,
            "capped": True,
        }

    return simple_answer, agentic_answer


def compute_scorecard(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    # Expects per-row method dict with: retrieved_ids, answer, expected, answerable, category.
    answerable = [r for r in rows if r["answerable"]]
    distractors = [r for r in rows if not r["answerable"]]

    def any_hit(expected: set[str], ids: list[str]) -> Optional[bool]:
        return (any(cid in expected for cid in ids) if expected else None)

    def cite_hit(expected: set[str], answer: str) -> Optional[bool]:
        if not expected:
            return None
        return any(f"[{cid}]" in answer for cid in expected)

    retrieval_hits = [any_hit(r["expected"], r[method]["retrieved_ids"]) for r in answerable]
    citation_hits = [cite_hit(r["expected"], r[method]["answer"]) for r in answerable]
    refusal_hits = [looks_like_refusal(r[method]["answer"]) for r in distractors]

    def mean_bool(xs: list[Optional[bool]]) -> float:
        vals = [x for x in xs if x is not None]
        return float(sum(vals)) / float(len(vals)) if vals else float("nan")

    return {
        "retrieval_hit_rate": mean_bool(retrieval_hits),
        "answer_citation_hit_rate": mean_bool(citation_hits),
        "distractor_refusal_correctness": float(sum(refusal_hits)) / float(len(refusal_hits)) if refusal_hits else float("nan"),
    }


def write_report_md(
    path: Path,
    *,
    run_ts: str,
    eval_path: Path,
    rows: list[dict[str, Any]],
    scorecards: dict[str, dict[str, Any]],
    costs: dict[str, dict[str, Any]],
) -> None:
    def pct(x: float) -> str:
        return "n/a" if (x != x) else f"{x*100:.1f}%"

    lines: list[str] = []
    lines.append("# Paired Eval Report (OKF vs Flat RAG vs Agentic RAG)")
    lines.append("")
    lines.append(f"- Run timestamp (UTC): `{run_ts}`")
    lines.append(f"- Eval set: `{eval_path.as_posix()}`")
    lines.append(f"- Models: generation `{GEN_MODEL}`, embeddings `{EMBED_MODEL}`")
    lines.append(f"- Dataset: `{KAGGLE_DATASET}` ({KAGGLE_URL})")
    lines.append("")
    lines.append("## What Is Compared")
    lines.append("")
    lines.append("- Same 14 questions, run in one execution.")
    lines.append("- OKF method uses `bundle/` concepts (Notebook 1-style).")
    lines.append("- Flat methods regenerate a notebook-2-style flat corpus and ingest into ChromaDB.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Small eval set; a single question can swing percentages noticeably.")
    lines.append("- Local model variance can change answers/citations between runs.")
    lines.append("- This is a hands-on architectural comparison, not a statistically-powered benchmark.")
    lines.append("")
    lines.append("## Aggregate Scorecard")
    lines.append("")
    lines.append("| Method | Retrieval hit rate | Answer citation-hit | Distractor refusal correctness | Avg calls / question | Avg searches / question |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for method, label in [
        ("okf", "OKF RAG (FAISS + hybrid rerank)"),
        ("simple", "Flat Simple RAG (ChromaDB)"),
        ("agentic", "Flat Agentic RAG (tool-calling)"),
    ]:
        sc = scorecards[method]
        c = costs[method]
        lines.append(
            f"| {label} | {pct(sc['retrieval_hit_rate'])} | {pct(sc['answer_citation_hit_rate'])} | "
            f"{pct(sc['distractor_refusal_correctness'])} | {c['avg_calls_per_question']:.2f} | {c['avg_searches_per_question']:.2f} |"
        )
    lines.append("")
    lines.append("## Per-Question Side-by-Side Answers")
    lines.append("")
    for i, r in enumerate(rows, 1):
        lines.append(f"### Q{i}. {r['q']}")
        lines.append("")
        lines.append(f"- Category: `{r['category']}`")
        lines.append(f"- Answerable: `{r['answerable']}`")
        lines.append(f"- Expected concept id(s): `{sorted(r['expected'])}`")
        lines.append("")
        for method, label, key in [
            ("okf", "OKF RAG", "okf"),
            ("simple", "Flat Simple RAG", "simple"),
            ("agentic", "Flat Agentic RAG", "agentic"),
        ]:
            m = r[key]
            lines.append(f"#### {label}")
            lines.append("")
            lines.append(f"- Retrieved ids: `{m['retrieved_ids']}`")
            if key == "okf":
                lines.append(f"- Final context ids: `{m['final_context_ids']}`")
            if key == "agentic":
                lines.append(f"- Searches: `{len(m['searches'])}`; Iterations: `{m['iterations']}`; Capped: `{m['capped']}`")
            lines.append(f"- Citations in answer: `{extract_bracket_citations(m['answer'])}`")
            lines.append("")
            lines.append("```text")
            lines.append(m["answer"].strip())
            lines.append("```")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO", format="<level>{message}</level>\n")

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("eval/eval_set_v1.yaml"))
    ap.add_argument("--bundle-root", type=Path, default=Path("bundle"))
    ap.add_argument("--out-json", type=Path, default=Path("artifacts/paired_eval_results.json"))
    ap.add_argument("--out-md", type=Path, default=Path("artifacts/paired_eval_report.md"))
    ap.add_argument("--snapshot-md", type=Path, default=Path("reports/paired_eval_report.md"))
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--okf-k-retrieve", type=int, default=8)
    ap.add_argument("--okf-k-final", type=int, default=4)
    ap.add_argument("--okf-k-raw-guaranteed", type=int, default=2)
    ap.add_argument("--agentic-max-iterations", type=int, default=4)
    args = ap.parse_args()

    run_ts = utc_now_iso()
    counter = CallCounter()

    # Preconditions
    local_models = {m["model"] for m in ollama.list()["models"]}
    for required in (GEN_MODEL, EMBED_MODEL):
        if not model_available(required, local_models):
            raise RuntimeError(f"Missing local model {required!r}. Pull it with `ollama pull {required}`.")

    if not args.bundle_root.exists():
        raise RuntimeError(
            f"Missing bundle directory at {args.bundle_root}. Run `bash scripts/e2e.sh` first to generate it."
        )

    eval_items = load_eval_set(args.eval)
    logger.info(f"Loaded eval set: {len(eval_items)} questions from {args.eval}")

    # OKF concepts (from bundle)
    okf_concepts = load_okf_concepts(args.bundle_root)
    okf_ids = {c["id"] for c in okf_concepts}

    # Flat corpus (regenerated notebook-2-style)
    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    table_rename = {"product_category_name_translation": "category_translation"}

    def table_name_from_file(path: Path) -> str:
        name = re.sub(r"_dataset$", "", re.sub(r"^olist_", "", path.stem))
        return table_rename.get(name, name)

    tables = {table_name_from_file(f): pd.read_csv(f) for f in sorted(dataset_dir.glob("*.csv"))}
    profiles = {name: profile_table(name, df) for name, df in tables.items()}
    fk_candidates = detect_fk_candidates(tables)

    orders_df = tables["orders"].copy()
    for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        orders_df[col] = pd.to_datetime(orders_df[col], errors="coerce")

    payments_per_order = tables["order_payments"].groupby("order_id")["payment_value"].sum()
    delivered = orders_df[orders_df["order_status"] == "delivered"].dropna(
        subset=["order_delivered_customer_date", "order_estimated_delivery_date"]
    )
    late_mask = delivered["order_delivered_customer_date"] > delivered["order_estimated_delivery_date"]
    review_scores = tables["order_reviews"]["review_score"]
    cat_en = tables["products"].merge(tables["category_translation"], on="product_category_name", how="left")
    top_categories = (
        tables["order_items"]
        .merge(cat_en[["product_id", "product_category_name_english"]], on="product_id", how="left")
        .groupby("product_category_name_english")["order_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(5)
    )

    metrics_summary = {
        "avg_order_value": {
            "value": round(float(payments_per_order.mean()), 2),
            "unit": "BRL",
            "n": int(payments_per_order.shape[0]),
            "source_tables": ["order_payments"],
            "definition": "mean(sum(payment_value) grouped by order_id)",
        },
        "late_delivery_rate": {
            "value": round(float(late_mask.mean()) * 100, 2),
            "unit": "%",
            "n": int(len(delivered)),
            "source_tables": ["orders"],
            "definition": "mean(delivered_date > estimated_date) over delivered orders",
        },
        "avg_review_score": {
            "value": round(float(review_scores.mean()), 2),
            "unit": "stars (1-5)",
            "n": int(review_scores.notna().sum()),
            "source_tables": ["order_reviews"],
            "definition": "mean(review_score)",
        },
        "review_score_distribution": {
            "value": (review_scores.value_counts(normalize=True).sort_index() * 100).round(2).to_dict(),
            "unit": "% of reviews",
            "n": int(review_scores.notna().sum()),
            "source_tables": ["order_reviews"],
            "definition": "value_counts(review_score, normalize=True)",
        },
        "payment_type_distribution": {
            "value": (tables["order_payments"]["payment_type"].value_counts(normalize=True) * 100).round(2).to_dict(),
            "unit": "% of payments",
            "n": int(tables["order_payments"].shape[0]),
            "source_tables": ["order_payments"],
            "definition": "value_counts(payment_type, normalize=True)",
        },
        "top_product_categories": {
            "value": top_categories.to_dict(),
            "unit": "distinct orders",
            "n": int(top_categories.sum()),
            "source_tables": ["order_items", "products", "category_translation"],
            "definition": "nunique(order_id) grouped by product_category_name_english, top 5",
        },
    }

    flat_docs = build_flat_corpus(counter, tables, profiles, fk_candidates, metrics_summary)
    flat_ids = {d["id"] for d in flat_docs}

    # Validate eval ids exist in both corpora for answerable questions.
    missing_okf: set[str] = set()
    missing_flat: set[str] = set()
    for it in eval_items:
        for cid in it.expected:
            if cid and cid not in okf_ids:
                missing_okf.add(cid)
            if cid and cid not in flat_ids:
                missing_flat.add(cid)
    if missing_okf:
        raise RuntimeError(f"Eval set references concept ids missing from OKF bundle: {sorted(missing_okf)}")
    if missing_flat:
        raise RuntimeError(f"Eval set references ids missing from flat corpus: {sorted(missing_flat)}")

    okf_answer = build_okf_pipeline(counter, okf_concepts)
    simple_answer, agentic_answer = build_flat_pipelines(counter, flat_docs)

    rows: list[dict[str, Any]] = []
    costs: dict[str, list[dict[str, Any]]] = {"okf": [], "simple": [], "agentic": []}

    for it in eval_items:
        row: dict[str, Any] = {
            "q": it.q,
            "expected": it.expected,
            "answerable": it.answerable,
            "category": it.category,
        }

        before = counter.snapshot()
        okf = okf_answer(it.q, k_retrieve=args.okf_k_retrieve, k_final=args.okf_k_final, k_raw_guaranteed=args.okf_k_raw_guaranteed)
        okf_calls = counter.delta(before)
        costs["okf"].append({"calls": okf_calls, "searches": 1.0})
        row["okf"] = okf

        before = counter.snapshot()
        s = simple_answer(it.q, top_k=args.top_k)
        s_calls = counter.delta(before)
        costs["simple"].append({"calls": s_calls, "searches": 1.0})
        row["simple"] = s

        before = counter.snapshot()
        a = agentic_answer(it.q, max_iterations=args.agentic_max_iterations)
        a_calls = counter.delta(before)
        n_searches = float(len(a["searches"])) if it.answerable else float(len(a["searches"]))
        costs["agentic"].append({"calls": a_calls, "searches": n_searches})
        row["agentic"] = a

        rows.append(row)

    scorecards = {
        "okf": compute_scorecard(rows, "okf"),
        "simple": compute_scorecard(rows, "simple"),
        "agentic": compute_scorecard(rows, "agentic"),
    }

    costs_summary: dict[str, dict[str, Any]] = {}
    for k, items in costs.items():
        costs_summary[k] = {
            "avg_calls_per_question": float(np.mean([x["calls"] for x in items])),
            "avg_searches_per_question": float(np.mean([x["searches"] for x in items])),
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "run_ts_utc": run_ts,
                "eval_path": args.eval.as_posix(),
                "models": {"gen": GEN_MODEL, "embed": EMBED_MODEL},
                "rows": rows,
                "scorecards": scorecards,
                "costs": costs_summary,
            },
            indent=2,
            default=list,
        )
        + "\n",
        encoding="utf-8",
    )

    write_report_md(
        args.out_md,
        run_ts=run_ts,
        eval_path=args.eval,
        rows=rows,
        scorecards=scorecards,
        costs=costs_summary,
    )
    write_report_md(
        args.snapshot_md,
        run_ts=run_ts,
        eval_path=args.eval,
        rows=rows,
        scorecards=scorecards,
        costs=costs_summary,
    )

    logger.info(f"Wrote JSON: {args.out_json}")
    logger.info(f"Wrote report: {args.out_md}")
    logger.info(f"Wrote snapshot: {args.snapshot_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

