"""
Full evaluation runner against goldendataset.txt.

Runs every golden query through the RAG pipeline, computes per-query and
aggregate metrics (by category), and saves eval_report.json.

Usage:
    python eval.py [--top-k N]
"""
import sys, io, json, argparse
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING", colorize=True,
           format="{time:HH:mm:ss} | {level} | {message}")

from config import config
from embedder import embedder
from searcher import searcher
from reranker import reranker
from llm import llm_synthesizer
from metrics import compute_all


# ── Parse golden dataset ─────────────────────────────────────────────────────

class GoldenItem:
    def __init__(self, query, answer, keywords, category):
        self.query = query
        self.answer = answer
        self.keywords = keywords
        self.category = category


def parse_golden_dataset(path: str) -> list[GoldenItem]:
    namespace = {"GoldenItem": GoldenItem, "List": list}
    with open(path, encoding="utf-8") as f:
        src = f.read()
    exec(src, namespace)
    return namespace["GOLDEN_DATASET"]


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run_query(query: str, top_k: int) -> tuple[list[dict], str | None]:
    try:
        embedding = embedder.embed(query)
        chunks = searcher.search(embedding, top_k, None)
        if chunks:
            chunks = reranker.rerank(query, chunks)
        answer = llm_synthesizer.synthesize(query, chunks) if chunks else None
        return chunks, answer
    except Exception as e:
        logger.error(f"Pipeline error for query '{query[:50]}': {e}")
        return [], None


# ── Reporting ─────────────────────────────────────────────────────────────────

def _avg(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 4) if valid else None


def _print_divider(char="─", width=72):
    print(char * width)


def _print_aggregate(label: str, records: list[dict]):
    if not records:
        return
    rm = [r["metrics"].get("ranked_metrics", {}) for r in records]
    aq = [r["metrics"].get("answer_quality", {}) for r in records]

    k_key = next((k for k in (rm[0] or {}) if k.startswith("precision@")), None)
    p = _avg([m.get(k_key) for m in rm]) if k_key else None
    r = _avg([m.get(k_key.replace("precision", "recall")) for m in rm]) if k_key else None
    f = _avg([m.get(k_key.replace("precision", "f1")) for m in rm]) if k_key else None
    mrr = _avg([m.get("mrr") for m in rm])
    hit_key = next((k for k in (rm[0] or {}) if k.startswith("hit_rate@")), None)
    hit = _avg([m.get(hit_key) for m in rm]) if hit_key else None
    cov = _avg([a.get("keyword_coverage") for a in aq])

    parts = []
    if p is not None: parts.append(f"P@K={p:.3f}")
    if r is not None: parts.append(f"R@K={r:.3f}")
    if f is not None: parts.append(f"F1@K={f:.3f}")
    if mrr is not None: parts.append(f"MRR={mrr:.3f}")
    if hit is not None: parts.append(f"Hit@K={hit:.3f}")
    if cov is not None: parts.append(f"AnswerCov={cov:.3f}")

    print(f"  {label:<14}  n={len(records)}  |  " + "  ".join(parts) if parts else f"  {label}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG against golden dataset")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    args = parser.parse_args()

    dataset_path = Path(__file__).parent / "goldendataset.txt"
    report_path  = Path(__file__).parent / "eval_report.json"

    items = parse_golden_dataset(str(dataset_path))
    config.validate()

    _print_divider("=")
    print(f"  EVALUATION  |  {len(items)} queries  |  top_k={args.top_k}  |  "
          f"db={config.DB_NAME}  |  table={config.TABLE_NAME}")
    _print_divider("=")

    results: list[dict] = []

    for i, item in enumerate(items, 1):
        is_adv = item.category == "adversarial"
        print(f"\n[{i:02d}] [{item.category}]  {item.query[:70]}")

        chunks, answer = run_query(item.query, args.top_k)

        metrics = compute_all(
            query=item.query,
            chunks=chunks,
            k=args.top_k,
            expected_keywords=item.keywords,
            is_adversarial=is_adv,
            answer=answer,
        )

        rm = metrics.get("ranked_metrics", {})
        aq = metrics.get("answer_quality", {})
        adv_ok = metrics.get("adversarial_ok", True)

        # Per-query summary line
        if is_adv:
            status = "PASS" if adv_ok else "FAIL"
            print(f"     chunks={len(chunks)}  adversarial_robustness={status}")
        else:
            k_key   = next((k for k in rm if k.startswith("precision@")), None)
            hit_key = next((k for k in rm if k.startswith("hit_rate@")),   None)
            p_val   = f"{rm[k_key]:.3f}" if k_key else "n/a"
            r_val   = f"{rm[k_key.replace('precision','recall')]:.3f}" if k_key else "n/a"
            mrr_val = f"{rm['mrr']:.3f}" if "mrr" in rm else "n/a"
            hit_val = f"{rm[hit_key]:.3f}" if hit_key else "n/a"
            cov_val = f"{aq['keyword_coverage']:.3f}" if aq.get("keyword_coverage") is not None else "n/a"
            matched = aq.get("matched", [])
            missing = aq.get("missing", [])
            print(f"     chunks={len(chunks)}")
            print(f"     P@K={p_val}  R@K={r_val}  MRR={mrr_val}  Hit@K={hit_val}")
            print(f"     AnswerCov={cov_val}  matched={matched}  missing={missing}")

        results.append({
            "query": item.query,
            "category": item.category,
            "expected_keywords": item.keywords,
            "chunks_retrieved": len(chunks),
            "answer": answer,
            "metrics": metrics,
        })

    # ── Aggregate by category ─────────────────────────────────────────────────
    print(f"\n")
    _print_divider("=")
    print("  AGGREGATE METRICS BY CATEGORY")
    _print_divider()

    categories = ["factual", "procedural", "comparative", "adversarial"]
    all_non_adv = [r for r in results if r["category"] != "adversarial"]
    all_adv     = [r for r in results if r["category"] == "adversarial"]

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        if cat == "adversarial":
            pass_count = sum(1 for r in cat_results if r["metrics"].get("adversarial_ok", False))
            print(f"  {'adversarial':<14}  n={len(cat_results)}  |  "
                  f"robustness={pass_count}/{len(cat_results)} PASS")
        else:
            _print_aggregate(cat, cat_results)

    _print_divider()
    _print_aggregate("OVERALL (non-adv)", all_non_adv)
    _print_divider("=")

    # ── Save report ───────────────────────────────────────────────────────────
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nReport saved → {report_path}\n")


if __name__ == "__main__":
    main()
