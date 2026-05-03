import argparse
import json
import subprocess
import sys
import io

# Force UTF-8 stdout/stderr on Windows to handle emoji output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from loguru import logger

# Configure loguru before any other imports
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=True,
)

from config import config
from embedder import embedder
from searcher import searcher
from reranker import reranker
from llm import llm_synthesizer


def _print_banner(text: str):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def _print_results(query: str, chunks: list[dict], answer: str | None):
    _print_banner(f"pginjection Results — Query: {query}")

    if answer:
        print("\n📝  SYNTHESIZED ANSWER\n")
        print(answer)
    else:
        print("\n⚠️   LLM unavailable — showing retrieved chunks only")

    print(f"\n📊  TOP {len(chunks)} CHUNKS (post-reranking)\n")
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("rerank_score", chunk.get("score", 0.0))
        meta = chunk.get("metadata", {})
        text_preview = chunk["chunk_text"][:300]
        ellipsis = "..." if len(chunk["chunk_text"]) > 300 else ""
        print(f"  [{i}]  Score: {score:.4f}", end="")
        if meta:
            meta_str = "  |  " + "  ".join(f"{k}={v}" for k, v in meta.items())
            print(meta_str, end="")
        print(f"\n       {text_preview}{ellipsis}\n")

    print("=" * 70)


def run(query: str, top_k: int, filters: dict | None, debug: bool):
    logger.info(
        f"[MAIN] Starting pginjection pipeline — "
        f"query='{query[:80]}{'...' if len(query) > 80 else ''}', "
        f"top_k={top_k}, filters={filters}, debug={debug}"
    )

    # Validate config
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"❌ [MAIN] Config validation failed: {e}")
        sys.exit(1)

    # Step 1 — Embed query (hard fail)
    try:
        embedding = embedder.embed(query)
    except Exception as e:
        logger.error(f"❌ [MAIN] Embedding failed — pipeline cannot continue: {e}")
        sys.exit(1)

    # Step 2 — Vector search (hard fail)
    try:
        chunks = searcher.search(embedding, top_k, filters)
    except Exception as e:
        logger.error(f"❌ [MAIN] Database search failed — pipeline cannot continue: {e}")
        sys.exit(1)

    if not chunks:
        logger.warning("⚠️  [MAIN] No chunks found above similarity threshold")
        print("\nNo relevant results found for your query.")
        return

    # Step 3 — Rerank (graceful degradation)
    chunks = reranker.rerank(query, chunks)
    logger.info(f"✅ [MAIN] Reranking complete — {len(chunks)} chunks retained")

    # Step 4 — LLM synthesis (graceful degradation)
    answer = llm_synthesizer.synthesize(query, chunks)
    if answer:
        logger.info("✅ [MAIN] Final answer synthesized successfully")
    else:
        logger.warning("⚠️  [MAIN] LLM synthesis unavailable — returning chunks only")

    # Step 5 — Print results
    _print_results(query, chunks, answer)

    # Step 6 — Save debug artefact and launch server if requested
    if debug:
        debug_data = {
            "query": query,
            "filters": filters or {},
            "top_k": top_k,
            "chunks": chunks,
            "answer": answer,
        }
        try:
            with open("debug_results.json", "w", encoding="utf-8") as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False, default=str)
            logger.info("✅ [MAIN] Debug results saved to debug_results.json")

            subprocess.Popen([sys.executable, "debug_server.py"])
            url = f"http://{config.DEBUG_API_HOST}:{config.DEBUG_API_PORT}"
            logger.info(f"✅ [MAIN] Debug server launching at {url}")
            print(f"\n🔍  Debug table: {url}")
        except Exception as e:
            logger.error(f"❌ [MAIN] Failed to launch debug server: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="pginjection — PostgreSQL RAG search with reranking and LLM synthesis"
    )
    parser.add_argument("--query", required=True, help="Natural language search query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {config.DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--filters",
        type=str,
        default=None,
        help='JSON metadata filters, e.g. \'{"category": "research", "date": "2023"}\'',
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save results to debug_results.json and launch FastAPI debug table",
    )
    args = parser.parse_args()

    filters = None
    if args.filters:
        try:
            filters = json.loads(args.filters)
        except json.JSONDecodeError as e:
            logger.error(f"❌ [MAIN] Invalid --filters JSON: {e}")
            sys.exit(1)

    run(args.query, args.top_k, filters, args.debug)


if __name__ == "__main__":
    main()
