"""SQ4 component test: measures whether RagChain's semantic cache preserves
answer relevance for paraphrased queries. For each base/paraphrase pair, the
cache is populated with the base query's response, then the paraphrase's
cached response is compared against a freshly generated response for the
same paraphrase (cache off) via embedding cosine similarity.
"""

import math
import pytest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL
from app.rag_chain import RagChain
from rag.semantic_cache import SemanticCache
from rag.embedding_cache import make_cached_embeddings
from scripts.benchmark_caching import BASE_QUERIES, PARAPHRASES
from tests.component.conftest import results

RELEVANCE_THRESHOLD = 0.85


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def get_response(cache: SemanticCache | None, query: str) -> str:
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.2, max_tokens=512, streaming=True)
    chain = RagChain(llm=llm, cache=cache)
    stream, _ = chain.run(
        user_input=query,
        history=[{"role": "user", "content": query}],
        temperature=0.2,
        max_tokens=512,
        k=4,
    )
    return "".join(c.content for c in stream)


@pytest.mark.integration
@pytest.mark.live_api
@pytest.mark.parametrize("base_query,paraphrase", list(zip(BASE_QUERIES, PARAPHRASES)))
def test_cached_response_is_semantically_relevant_to_fresh_response(base_query, paraphrase):
    cache = SemanticCache(embeddings=make_cached_embeddings())
    get_response(cache, base_query)

    cached_response = get_response(cache, paraphrase)
    fresh_response = get_response(None, paraphrase)

    embedder = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    cached_vector = embedder.embed_query(cached_response)
    fresh_vector = embedder.embed_query(fresh_response)
    similarity = cosine_similarity(cached_vector, fresh_vector)

    results.append(
        {
            "base_query": base_query,
            "paraphrase": paraphrase,
            "cached_response": cached_response,
            "fresh_response": fresh_response,
            "similarity": similarity,
        }
    )

    assert similarity >= RELEVANCE_THRESHOLD, (
        f"Cached response for paraphrase {paraphrase!r} diverged from a fresh response "
        f"(similarity={similarity:.4f}, threshold={RELEVANCE_THRESHOLD})"
    )
