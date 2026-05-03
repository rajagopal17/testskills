"""
Retrieval evaluation metrics for pginjection debug mode.

Two modes:
  1. Score-based stats  — always available (no ground truth needed)
  2. Ranked metrics     — requires ground_truth.json in the skill directory

ground_truth.json format:
  {
    "What is attention?": ["chunk_id_1", "chunk_id_2"],
    "Explain transformers": ["chunk_id_3"]
  }

Chunks must have an "id" field for ground-truth matching.
Computed metrics: Precision@K, Recall@K, F1@K, MRR, NDCG@K, Hit Rate@K.
"""

import math
import json
from pathlib import Path
from loguru import logger

GROUND_TRUTH_FILE = Path(__file__).parent / "ground_truth.json"


# ── Score-based statistics (always available) ────────────────────────────────

def score_stats(chunks: list[dict]) -> dict:
    """Descriptive stats over similarity and rerank scores."""
    if not chunks:
        return {}

    sim_scores = [c.get("score", 0.0) for c in chunks]
    rerank_scores = [c["rerank_score"] for c in chunks if "rerank_score" in c]

    def _stats(values: list[float], label: str) -> dict:
        n = len(values)
        if n == 0:
            return {}
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        return {
            f"{label}_count": n,
            f"{label}_min": round(min(values), 4),
            f"{label}_max": round(max(values), 4),
            f"{label}_mean": round(mean, 4),
            f"{label}_std": round(math.sqrt(variance), 4),
            f"{label}_gap_top2": round(values[0] - values[1], 4) if n >= 2 else None,
        }

    stats = {}
    stats.update(_stats(sim_scores, "similarity"))
    if rerank_scores:
        stats.update(_stats(rerank_scores, "rerank"))

    # Score correlation (Spearman rank correlation between similarity and rerank order)
    if rerank_scores and len(rerank_scores) == len(sim_scores):
        try:
            from scipy.stats import spearmanr
            corr, pval = spearmanr(sim_scores, rerank_scores)
            stats["rerank_similarity_spearman_r"] = round(float(corr), 4)
            stats["rerank_similarity_spearman_p"] = round(float(pval), 4)
        except Exception:
            pass

    return stats


# ── Ranked retrieval metrics (requires ground truth) ─────────────────────────

def _load_ground_truth() -> dict:
    if not GROUND_TRUTH_FILE.exists():
        return {}
    try:
        with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(
            f"✅ [METRICS] Loaded ground truth for {len(data)} queries "
            f"from {GROUND_TRUTH_FILE.name}"
        )
        return data
    except Exception as e:
        logger.error(f"❌ [METRICS] Failed to load ground_truth.json: {e}")
        return {}


def _precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k if k > 0 else 0.0


def _recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def _f1_at_k(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, rid in enumerate(retrieved_ids[:k], 1)
        if rid in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return 1.0 if any(rid in relevant_ids for rid in retrieved_ids[:k]) else 0.0


def ranked_metrics(query: str, chunks: list[dict], k: int | None = None) -> dict:
    """
    Compute P@K, R@K, F1@K, MRR, NDCG@K, Hit Rate@K for a single query.
    Returns empty dict if ground_truth.json is absent or query not found.
    """
    ground_truth = _load_ground_truth()
    if not ground_truth or query not in ground_truth:
        logger.info(
            "[METRICS] No ground truth found for this query — "
            "ranked metrics unavailable. Add entries to ground_truth.json to enable."
        )
        return {}

    relevant_ids = set(ground_truth[query])
    retrieved_ids = [str(c.get("id", c.get("chunk_id", ""))) for c in chunks]
    k = k or len(chunks)

    precision = _precision_at_k(retrieved_ids, relevant_ids, k)
    recall = _recall_at_k(retrieved_ids, relevant_ids, k)
    f1 = _f1_at_k(precision, recall)
    mrr = _mrr(retrieved_ids, relevant_ids)
    ndcg = _ndcg_at_k(retrieved_ids, relevant_ids, k)
    hit_rate = _hit_rate_at_k(retrieved_ids, relevant_ids, k)

    results = {
        f"precision@{k}": round(precision, 4),
        f"recall@{k}": round(recall, 4),
        f"f1@{k}": round(f1, 4),
        "mrr": round(mrr, 4),
        f"ndcg@{k}": round(ndcg, 4),
        f"hit_rate@{k}": round(hit_rate, 4),
        "relevant_total": len(relevant_ids),
        "retrieved_total": len(retrieved_ids),
    }

    logger.info(
        f"✅ [METRICS] Ranked metrics computed — "
        f"P@{k}={precision:.3f}, R@{k}={recall:.3f}, "
        f"MRR={mrr:.3f}, NDCG@{k}={ndcg:.3f}, Hit@{k}={hit_rate:.1f}"
    )
    return results


def answer_quality(answer: str | None, expected_keywords: list[str]) -> dict:
    """
    Measure answer quality by keyword coverage.
    Returns fraction of expected keywords found in the synthesized answer.
    """
    if not answer or not expected_keywords:
        return {"keyword_coverage": None, "matched": [], "missing": expected_keywords or []}
    answer_lower = answer.lower()
    matched = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    coverage = round(len(matched) / len(expected_keywords), 4) if expected_keywords else None
    logger.info(
        f"✅ [METRICS] Answer quality — coverage={coverage} "
        f"({len(matched)}/{len(expected_keywords)} keywords matched)"
    )
    return {"keyword_coverage": coverage, "matched": matched, "missing": missing}


def adversarial_check(chunks: list[dict], is_adversarial: bool) -> bool:
    """
    For adversarial queries (expected answer = UNANSWERABLE), the RAG should
    return 0 chunks. Returns True if behaviour is correct.
    """
    if not is_adversarial:
        return True
    correct = len(chunks) == 0
    if not correct:
        logger.warning(
            f"⚠️  [METRICS] Adversarial robustness failure — "
            f"returned {len(chunks)} chunk(s) for an unanswerable query"
        )
    return correct


def compute_all(
    query: str,
    chunks: list[dict],
    k: int | None = None,
    expected_keywords: list[str] | None = None,
    is_adversarial: bool = False,
    answer: str | None = None,
) -> dict:
    """Compute all available metrics and return as a single dict."""
    stats = score_stats(chunks)
    ranked = ranked_metrics(query, chunks, k)
    quality = answer_quality(answer, expected_keywords or [])
    adv_ok = adversarial_check(chunks, is_adversarial)
    return {
        "score_stats": stats,
        "ranked_metrics": ranked,
        "answer_quality": quality,
        "adversarial_ok": adv_ok,
    }
