"""SQ3 benchmark: measures latency and API cost for each caching strategy in
isolation and combined, compared to the uncached baseline.

Part 1 (query-time): runs 30 queries (10 base questions, each repeated as an
exact duplicate and a paraphrase) through four configurations: baseline (no
caching), embedding cache only, semantic cache only, and both. Requires
Breach-Report-2024.pdf to already be ingested into ChromaDB.

Part 2 (ingestion-time): re-ingests Breach-Report-2024.pdf into a fresh,
temporary vector store twice, cold (embedding cache empty) vs warm (embedding
cache already populated from the cold run), to measure the corpus-rebuild
benefit SQ1 identified as embedding caching's primary use case. Fully
self-contained in temp directories, does not touch the real ChromaDB or
embedding cache.

Run with: python scripts/benchmark_caching.py
"""

import json
import os
import tempfile
import time
from unittest.mock import patch

import tiktoken
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL
from app.rag_chain import RagChain
from rag.document_loader import BasicPdfLoader
from rag.semantic_cache import SemanticCache
from rag.embedding_cache import CachedEmbeddings, make_cached_embeddings

GPT4O_INPUT_PRICE_PER_M = 2.50
GPT4O_OUTPUT_PRICE_PER_M = 10.00
EMBEDDING_PRICE_PER_M = 0.020

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "benchmark_results")
KNOWLEDGE_BASE_PDF = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base", "Breach-Report-2024.pdf")

BASE_QUERIES = [
    "What is the average cost of a data breach in 2024?",
    "What are the most common initial attack vectors for data breaches?",
    "How long does it typically take to identify and contain a data breach?",
    "How does using AI and automation in security affect breach costs?",
    "Do companies raise prices after experiencing a data breach?",
    "What is the average cost of an extortion or ransomware attack?",
    "Which industries have the highest data breach costs?",
    "What factors increase the cost of a data breach the most?",
    "How much do regulatory fines add to the cost of a data breach?",
    "What is a mega breach and how much does it cost?",
]

PARAPHRASES = [
    "How much does a data breach cost on average in 2024?",
    "What are the top ways attackers initially breach an organization?",
    "What is the average time to detect and contain a breach?",
    "Does security AI and automation reduce the cost of a breach?",
    "Do businesses increase prices for customers after a breach?",
    "How expensive are ransomware and extortion attacks on average?",
    "What industry experiences the most expensive data breaches?",
    "What are the biggest cost-increasing factors for a breach?",
    "What role do regulatory fines play in total breach costs?",
    "How costly are mega breaches compared to typical breaches?",
]

QUERIES = BASE_QUERIES + BASE_QUERIES + PARAPHRASES

CONFIGURATIONS = [
    ("baseline", False, False),
    ("embedding_cache_only", True, False),
    ("semantic_cache_only", False, True),
    ("both_caches", True, True),
]

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def run_query(query: str, cache: SemanticCache | None, embedding_cache_enabled: bool, seen_texts: set) -> dict:
    is_semantic_hit = cache is not None and cache.check(query) is not None
    is_embedding_hit = embedding_cache_enabled and query in seen_texts
    seen_texts.add(query)

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.2,
        max_tokens=512,
        streaming=True,
        stream_usage=True,
    )
    chain = RagChain(llm=llm, cache=cache)
    start = time.perf_counter()
    stream, docs = chain.run(
        user_input=query,
        history=[{"role": "user", "content": query}],
        temperature=0.2,
        max_tokens=512,
        k=4,
    )
    chunks = list(stream)
    elapsed = time.perf_counter() - start
    full_text = "".join(c.content for c in chunks)

    embed_cost = 0.0 if is_embedding_hit else count_tokens(query) / 1_000_000 * EMBEDDING_PRICE_PER_M

    if is_semantic_hit:
        input_tokens = 0
        output_tokens = 0
        generation_cost = 0.0
    else:
        usage = next((getattr(c, "usage_metadata", None) for c in chunks if getattr(c, "usage_metadata", None)), None)
        input_tokens = usage["input_tokens"] if usage else count_tokens(query)
        output_tokens = usage["output_tokens"] if usage else count_tokens(full_text)
        generation_cost = (
            input_tokens / 1_000_000 * GPT4O_INPUT_PRICE_PER_M
            + output_tokens / 1_000_000 * GPT4O_OUTPUT_PRICE_PER_M
        )

    return {
        "query": query,
        "semantic_cache_hit": is_semantic_hit,
        "embedding_cache_hit": is_embedding_hit,
        "latency_seconds": elapsed,
        "cost_usd": embed_cost + generation_cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "response_chars": len(full_text),
        "docs_retrieved": len(docs),
    }


def run_pipeline(use_embedding_cache: bool, use_semantic_cache: bool) -> list[dict]:
    cache = SemanticCache(embeddings=make_cached_embeddings()) if use_semantic_cache else None
    seen_texts = set()

    if use_embedding_cache:
        return [run_query(q, cache, True, seen_texts) for q in QUERIES]

    plain_embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    with patch("rag.store.make_cached_embeddings", lambda: plain_embeddings):
        return [run_query(q, cache, False, seen_texts) for q in QUERIES]


def summarize(label: str, results: list[dict]) -> dict:
    total_latency = sum(r["latency_seconds"] for r in results)
    total_cost = sum(r["cost_usd"] for r in results)
    semantic_hits = sum(1 for r in results if r["semantic_cache_hit"])
    embedding_hits = sum(1 for r in results if r["embedding_cache_hit"])
    return {
        "label": label,
        "queries": len(results),
        "semantic_cache_hits": semantic_hits,
        "semantic_hit_rate_pct": round(semantic_hits / len(results) * 100, 1),
        "embedding_cache_hits": embedding_hits,
        "embedding_hit_rate_pct": round(embedding_hits / len(results) * 100, 1),
        "total_latency_seconds": round(total_latency, 3),
        "avg_latency_seconds": round(total_latency / len(results), 3),
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_usd": round(total_cost / len(results), 6),
    }


def run_query_time_benchmark() -> dict:
    query_results = {}
    for label, use_embedding, use_semantic in CONFIGURATIONS:
        print(f"Running {len(QUERIES)} queries: {label}...")
        results = run_pipeline(use_embedding_cache=use_embedding, use_semantic_cache=use_semantic)
        query_results[label] = {"summary": summarize(label, results), "queries": results}

    baseline_summary = query_results["baseline"]["summary"]
    for label, _, _ in CONFIGURATIONS[1:]:
        s = query_results[label]["summary"]
        s["latency_reduction_pct_vs_baseline"] = round(
            (1 - s["avg_latency_seconds"] / baseline_summary["avg_latency_seconds"]) * 100, 1
        )
        s["cost_reduction_pct_vs_baseline"] = round(
            (1 - s["avg_cost_usd"] / baseline_summary["avg_cost_usd"]) * 100, 1
        )

    return query_results


def run_ingestion_benchmark(chunk_size: int = 1000, chunk_overlap: int = 150) -> dict:
    loader = BasicPdfLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = loader.load(KNOWLEDGE_BASE_PDF)
    embed_tokens = sum(count_tokens(c.page_content) for c in chunks)

    from langchain_classic.storage import LocalFileStore
    import shutil

    cache_dir = tempfile.mkdtemp()
    chroma_cold = tempfile.mkdtemp()
    chroma_warm = tempfile.mkdtemp()
    chroma_uncached = tempfile.mkdtemp()

    try:
        underlying = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        cached_embedder = CachedEmbeddings(
            embeddings=underlying, store=LocalFileStore(cache_dir), namespace=OPENAI_EMBEDDING_MODEL
        )

        start = time.perf_counter()
        Chroma.from_documents(documents=chunks, embedding=cached_embedder, persist_directory=chroma_cold)
        cold_seconds = time.perf_counter() - start

        start = time.perf_counter()
        Chroma.from_documents(documents=chunks, embedding=cached_embedder, persist_directory=chroma_warm)
        warm_seconds = time.perf_counter() - start

        plain_embedder = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        start = time.perf_counter()
        Chroma.from_documents(documents=chunks, embedding=plain_embedder, persist_directory=chroma_uncached)
        uncached_seconds = time.perf_counter() - start
    finally:
        for d in (cache_dir, chroma_cold, chroma_warm, chroma_uncached):
            shutil.rmtree(d, ignore_errors=True)

    estimated_cost_per_full_embed = embed_tokens / 1_000_000 * EMBEDDING_PRICE_PER_M

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": len(chunks),
        "embed_tokens": embed_tokens,
        "cold_seconds": round(cold_seconds, 3),
        "warm_seconds": round(warm_seconds, 3),
        "uncached_seconds": round(uncached_seconds, 3),
        "estimated_cost_usd_cold_or_uncached": round(estimated_cost_per_full_embed, 6),
        "estimated_cost_usd_warm": 0.0,
        "latency_reduction_pct_warm_vs_cold": round((1 - warm_seconds / cold_seconds) * 100, 1),
    }


def main():
    print("=== Part 1: query-time RAG benchmark (4 configurations) ===")
    query_results = run_query_time_benchmark()

    print("\n--- Query-time results ---")
    for label, _, _ in CONFIGURATIONS:
        s = query_results[label]["summary"]
        line = f"{label}: avg latency {s['avg_latency_seconds']}s, avg cost ${s['avg_cost_usd']}"
        if label != "baseline":
            line += f", latency reduction {s['latency_reduction_pct_vs_baseline']}%, cost reduction {s['cost_reduction_pct_vs_baseline']}%"
        print(line)

    print("\n=== Part 2a: ingestion-time embedding cache benchmark (default chunk_size=1000, realistic app setting) ===")
    ingestion_default = run_ingestion_benchmark(chunk_size=1000, chunk_overlap=150)
    print(f"Chunks: {ingestion_default['num_chunks']}")
    print(f"Cold (cache empty): {ingestion_default['cold_seconds']}s")
    print(f"Warm (cache populated, corpus rebuild): {ingestion_default['warm_seconds']}s")
    print(f"Uncached (no caching at all): {ingestion_default['uncached_seconds']}s")
    print(f"Latency reduction, warm vs cold: {ingestion_default['latency_reduction_pct_warm_vs_cold']}%")

    print("\n=== Part 2b: ingestion-time benchmark, exploratory (chunk_size=50, below the app's normal range, forces multiple batched API requests) ===")
    ingestion_small_chunks = run_ingestion_benchmark(chunk_size=50, chunk_overlap=10)
    print(f"Chunks: {ingestion_small_chunks['num_chunks']}")
    print(f"Cold (cache empty): {ingestion_small_chunks['cold_seconds']}s")
    print(f"Warm (cache populated, corpus rebuild): {ingestion_small_chunks['warm_seconds']}s")
    print(f"Uncached (no caching at all): {ingestion_small_chunks['uncached_seconds']}s")
    print(f"Latency reduction, warm vs cold: {ingestion_small_chunks['latency_reduction_pct_warm_vs_cold']}%")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "sq3_results.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "query_time": query_results,
                "ingestion_time_default_chunk_size": ingestion_default,
                "ingestion_time_small_chunk_size_exploratory": ingestion_small_chunks,
            },
            f,
            indent=2,
        )
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
