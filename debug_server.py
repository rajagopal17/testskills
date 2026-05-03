"""
pginjection Debug Server — explicit debug mode only.

Launch:  python debug_server.py
         (or automatically via main.py --debug)

Reads debug_results.json and serves an HTML page with:
  - Query info and synthesized answer
  - Score statistics panel
  - Retrieval accuracy metrics (P@K, R@K, F1@K, MRR, NDCG@K, Hit Rate@K)
  - Answer quality panel  (keyword coverage — requires goldendataset.txt)
  - Adversarial robustness panel (requires goldendataset.txt)
  - Ranked chunks table
"""

import json
import sys
from pathlib import Path
from loguru import logger

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from config import config
from metrics import compute_all

app = FastAPI(title="pginjection Debug", docs_url=None, redoc_url=None)

DEBUG_FILE          = Path(__file__).parent / "debug_results.json"
GOLDEN_DATASET_FILE = Path(__file__).parent / "goldendataset.txt"


# ── Golden dataset loader ────────────────────────────────────────────────────

class GoldenItem:
    def __init__(self, query, answer, keywords, category):
        self.query    = query
        self.answer   = answer
        self.keywords = keywords
        self.category = category


def _load_golden_item(query: str) -> "GoldenItem | None":
    if not GOLDEN_DATASET_FILE.exists():
        return None
    try:
        namespace = {"GoldenItem": GoldenItem, "List": list}
        with open(GOLDEN_DATASET_FILE, encoding="utf-8") as f:
            exec(f.read(), namespace)
        items = namespace.get("GOLDEN_DATASET", [])
        return next((item for item in items if item.query == query), None)
    except Exception as e:
        logger.warning(f"[DEBUG_SERVER] Could not load golden dataset: {e}")
        return None


# ── Debug results loader ─────────────────────────────────────────────────────

def _load() -> dict:
    if not DEBUG_FILE.exists():
        logger.warning(f"[DEBUG_SERVER] {DEBUG_FILE.name} not found — run main.py --debug first")
        return {}
    try:
        with open(DEBUG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"[DEBUG_SERVER] Loaded {len(data.get('chunks', []))} chunks")
        return data
    except Exception as e:
        logger.error(f"[DEBUG_SERVER] Failed to read {DEBUG_FILE.name}: {e}")
        return {}


# ── HTML helpers ─────────────────────────────────────────────────────────────

def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )


def _stat_card(label: str, value: str, color: str = "#6366f1") -> str:
    return f"""
    <div class="stat-card" style="border-top:3px solid {color}">
      <div class="stat-value">{value}</div>
      <div class="stat-label">{label}</div>
    </div>"""


def _kw_badge(word: str, matched: bool) -> str:
    bg  = "#dcfce7" if matched else "#fee2e2"
    fg  = "#15803d" if matched else "#b91c1c"
    ico = "✓" if matched else "✗"
    return (
        f'<span class="kw-badge" style="background:{bg};color:{fg}">'
        f'{ico} {_esc(word)}</span>'
    )


# ── Metrics section renderer ─────────────────────────────────────────────────

def _render_score_stats(score_stats: dict) -> str:
    if not score_stats:
        return ""
    colour_map = {"similarity": "#6366f1", "rerank": "#10b981", "spearman": "#f59e0b"}

    def _col(key):
        for k, c in colour_map.items():
            if k in key:
                return c
        return "#6366f1"

    cards = "".join(
        _stat_card(k.replace("_", " ").title(), str(v), _col(k))
        for k, v in score_stats.items() if v is not None
    )
    return f"""
    <section class="metrics-section">
      <h2>&#x1F4C8; Score Statistics</h2>
      <p class="hint">Computed from similarity and rerank scores — no ground truth required.</p>
      <div class="stat-grid">{cards}</div>
    </section>"""


def _render_retrieval_accuracy(ranked: dict) -> str:
    ranked_colours = {
        "precision": "#6366f1", "recall": "#10b981", "f1": "#3b82f6",
        "mrr": "#f59e0b", "ndcg": "#8b5cf6", "hit_rate": "#ef4444",
        "total": "#64748b",
    }

    def _col(key):
        for k, c in ranked_colours.items():
            if k in key.lower():
                return c
        return "#6366f1"

    if not ranked:
        return """
    <section class="metrics-section metrics-dim">
      <h2>&#x1F3AF; Retrieval Accuracy Metrics</h2>
      <p class="hint"><strong>Unavailable</strong> — add
        <code>ground_truth.json</code> to enable P@K, R@K, F1@K, MRR, NDCG@K, Hit Rate@K.
      </p>
    </section>"""

    cards = "".join(
        _stat_card(
            k.replace("_", " ").replace("@", " @ ").upper(),
            f"{v:.4f}" if isinstance(v, float) else str(v),
            _col(k),
        )
        for k, v in ranked.items() if v is not None
    )
    rel   = ranked.get("relevant_total", "?")
    retr  = ranked.get("retrieved_total", "?")
    return f"""
    <section class="metrics-section">
      <h2>&#x1F3AF; Retrieval Accuracy Metrics</h2>
      <p class="hint">
        Against <code>ground_truth.json</code> &mdash;
        <strong>{rel}</strong> relevant chunks in DB,
        <strong>{retr}</strong> retrieved by RAG.
      </p>
      <div class="stat-grid">{cards}</div>
    </section>"""


def _render_answer_quality(aq: dict) -> str:
    coverage = aq.get("keyword_coverage")
    matched  = aq.get("matched", [])
    missing  = aq.get("missing", [])

    if coverage is None and not matched and not missing:
        return """
    <section class="metrics-section metrics-dim">
      <h2>&#x1F4AC; Answer Quality</h2>
      <p class="hint"><strong>Unavailable</strong> — query not found in
        <code>goldendataset.txt</code>. Keyword coverage cannot be computed.
      </p>
    </section>"""

    pct        = int((coverage or 0) * 100)
    bar_color  = "#22c55e" if pct >= 75 else "#f59e0b" if pct >= 40 else "#ef4444"
    badges     = "".join(_kw_badge(kw, True) for kw in matched)
    badges    += "".join(_kw_badge(kw, False) for kw in missing)
    total      = len(matched) + len(missing)

    return f"""
    <section class="metrics-section">
      <h2>&#x1F4AC; Answer Quality &mdash; Keyword Coverage</h2>
      <p class="hint">
        Fraction of expected keywords present in the synthesized answer
        (<strong>{len(matched)}/{total}</strong> matched).
      </p>
      <div class="coverage-bar-wrap">
        <div class="coverage-bar" style="width:{pct}%;background:{bar_color}"></div>
        <span class="coverage-pct" style="color:{bar_color}">{pct}%</span>
      </div>
      <div class="kw-list">{badges if badges else '<em style="color:#94a3b8">no keywords defined</em>'}</div>
    </section>"""


def _render_adversarial(adv_ok: bool, is_adversarial: bool, n_chunks: int) -> str:
    if not is_adversarial:
        return ""

    if adv_ok:
        return """
    <section class="metrics-section adv-pass">
      <h2>&#x1F6E1; Adversarial Robustness &mdash; <span style="color:#15803d">PASS</span></h2>
      <p class="hint">
        Correctly returned <strong>0 chunks</strong> for an unanswerable query.
        The RAG did not hallucinate context where none exists.
      </p>
    </section>"""
    else:
        return f"""
    <section class="metrics-section adv-fail">
      <h2>&#x1F6E1; Adversarial Robustness &mdash; <span style="color:#b91c1c">FAIL</span></h2>
      <p class="hint">
        Returned <strong>{n_chunks} chunk(s)</strong> for an unanswerable query.
        Consider raising <code>SIMILARITY_THRESHOLD</code> in <code>.env</code>
        to reduce false retrievals.
      </p>
    </section>"""


def _render_metrics_section(all_metrics: dict, is_adversarial: bool, n_chunks: int) -> str:
    score_stats = all_metrics.get("score_stats", {})
    ranked      = all_metrics.get("ranked_metrics", {})
    aq          = all_metrics.get("answer_quality", {})
    adv_ok      = all_metrics.get("adversarial_ok", True)

    return (
        _render_score_stats(score_stats)
        + _render_retrieval_accuracy(ranked)
        + _render_answer_quality(aq)
        + _render_adversarial(adv_ok, is_adversarial, n_chunks)
    )


# ── Page renderer ─────────────────────────────────────────────────────────────

def _render_page(data: dict) -> str:
    raw_query    = data.get("query", "")
    query        = _esc(raw_query)
    filters      = data.get("filters") or {}
    chunks       = data.get("chunks", [])
    answer       = data.get("answer", "")

    # Load golden metadata for this query (if available)
    golden = _load_golden_item(raw_query)
    expected_kws  = golden.keywords  if golden else []
    is_adversarial = golden.category == "adversarial" if golden else False

    all_metrics = compute_all(
        query           = raw_query,
        chunks          = chunks,
        k               = len(chunks),
        expected_keywords = expected_kws,
        is_adversarial  = is_adversarial,
        answer          = answer,
    )
    metrics_html = _render_metrics_section(all_metrics, is_adversarial, len(chunks))

    # Answer box
    answer_html = ""
    if answer:
        answer_html = f"""
        <section class="answer-box">
          <h2>&#x1F4DD; Synthesized Answer</h2>
          <div class="answer-text">{_esc(answer)}</div>
        </section>"""

    # Chunks table
    rows_html = ""
    for i, chunk in enumerate(chunks, 1):
        sim_score    = chunk.get("score", 0.0)
        rerank_score = chunk.get("rerank_score")
        rerank_str   = f"{rerank_score:.4f}" if rerank_score is not None else "—"
        chunk_id     = chunk.get("id", "")
        meta         = chunk.get("metadata", {})
        meta_html    = (
            " ".join(
                f'<span class="badge">{_esc(k)}: {_esc(str(v))}</span>'
                for k, v in meta.items()
            ) or "—"
        )
        text      = _esc(chunk.get("chunk_text", ""))
        row_class = (
            "row-high" if sim_score >= 0.85
            else "row-mid" if sim_score >= 0.70
            else "row-low"
        )
        id_badge = f'<span class="id-badge">id:{_esc(str(chunk_id))}</span>' if chunk_id else ""
        rows_html += f"""
        <tr class="{row_class}">
          <td class="col-sno">{i}{id_badge}</td>
          <td class="col-text">{text}</td>
          <td class="col-score">{sim_score:.4f}</td>
          <td class="col-score">{rerank_str}</td>
          <td class="col-meta">{meta_html}</td>
        </tr>"""

    filters_str  = json.dumps(filters) if filters else "none"
    category_tag = (
        f'<span class="cat-badge cat-{golden.category}">{golden.category}</span>'
        if golden else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>pginjection Debug</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: #f1f5f9; color: #1e293b;
      padding: 2rem; font-size: 14px;
    }}
    h1 {{ font-size: 1.6rem; color: #0f172a; margin-bottom: 0.25rem; }}
    h2 {{ font-size: 1.05rem; color: #334155; margin-bottom: 0.5rem; }}
    .subtitle {{ color: #64748b; font-size: 0.85rem; margin-bottom: 2rem; }}
    /* ── Query card ── */
    .query-card {{
      background: white; border-left: 4px solid #6366f1;
      border-radius: 6px; padding: 1rem 1.25rem;
      margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.07);
    }}
    .query-card .label {{ font-size: 0.75rem; text-transform: uppercase;
      letter-spacing:.05em; color:#94a3b8; margin-bottom:2px; }}
    .query-card .value {{ font-weight:600; color:#0f172a; }}
    /* ── Answer box ── */
    .answer-box {{
      background: #f0fdf4; border-left: 4px solid #22c55e;
      border-radius: 6px; padding: 1rem 1.25rem;
      margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.07);
    }}
    .answer-text {{ white-space: pre-wrap; line-height: 1.7; margin-top: 0.5rem; }}
    /* ── Metrics panels ── */
    .metrics-section {{
      background: white; border-radius: 8px;
      padding: 1.25rem; margin-bottom: 1.5rem;
      box-shadow: 0 1px 3px rgba(0,0,0,.07);
    }}
    .metrics-dim {{ opacity: 0.55; }}
    .adv-pass {{ border-left: 4px solid #22c55e; }}
    .adv-fail {{ border-left: 4px solid #ef4444; }}
    .hint {{
      font-size: 0.8rem; color: #64748b;
      margin-bottom: 1rem; margin-top: 0.25rem; line-height: 1.6;
    }}
    /* ── Stat grid ── */
    .stat-grid {{ display:flex; flex-wrap:wrap; gap:0.75rem; margin-top:0.75rem; }}
    .stat-card {{
      background: #f8fafc; border-radius: 6px;
      padding: 0.75rem 1rem; min-width: 130px;
      box-shadow: 0 1px 2px rgba(0,0,0,.05);
    }}
    .stat-value {{ font-size: 1.25rem; font-weight: 700; color: #0f172a; }}
    .stat-label {{ font-size: 0.7rem; text-transform: uppercase;
      letter-spacing:.05em; color:#94a3b8; margin-top:2px; }}
    /* ── Coverage bar ── */
    .coverage-bar-wrap {{
      position: relative; background: #e2e8f0;
      border-radius: 99px; height: 20px; margin: 0.75rem 0 0.5rem;
      overflow: hidden;
    }}
    .coverage-bar {{
      height: 100%; border-radius: 99px;
      transition: width .4s ease;
    }}
    .coverage-pct {{
      position: absolute; right: 8px; top: 1px;
      font-size: 0.8rem; font-weight: 700; line-height: 18px;
    }}
    /* ── Keyword badges ── */
    .kw-list {{ display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.75rem; }}
    .kw-badge {{
      display: inline-block; padding: 3px 10px;
      border-radius: 12px; font-size: 0.8rem; font-weight: 600;
    }}
    /* ── Chunks table ── */
    .table-section {{
      background: white; border-radius: 8px; overflow: hidden;
      box-shadow: 0 1px 3px rgba(0,0,0,.07); margin-bottom: 2rem;
    }}
    .table-header {{ padding: 1rem 1.25rem 0.75rem; border-bottom: 1px solid #e2e8f0; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{
      background: #6366f1; color: white;
      padding: 10px 14px; text-align: left;
      font-size: 0.8rem; text-transform: uppercase;
      letter-spacing:.05em; font-weight:600;
    }}
    td {{ padding: 10px 14px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }}
    tr:last-child td {{ border-bottom: none; }}
    .row-high td {{ background: #f0fdf4; }}
    .row-mid  td {{ background: #fefce8; }}
    .row-low  td {{ background: #fff7ed; }}
    .col-sno {{ width:55px; color:#94a3b8; font-weight:700; text-align:center; }}
    .col-text {{ max-width:520px; word-break:break-word; line-height:1.6; }}
    .col-score {{
      width:90px; font-weight:700; color:#6366f1;
      text-align:right; font-variant-numeric:tabular-nums;
    }}
    .col-meta {{ width:180px; }}
    .badge {{
      display:inline-block; padding:2px 7px;
      background:#e0e7ff; color:#4338ca;
      border-radius:12px; font-size:0.75rem;
      margin:2px 2px 2px 0;
    }}
    .id-badge {{
      display:block; font-size:0.65rem; color:#94a3b8;
      margin-top:2px; font-weight:400;
    }}
    /* ── Category badge ── */
    .cat-badge {{
      display:inline-block; padding:2px 9px;
      border-radius:12px; font-size:0.72rem; font-weight:600;
      margin-left:8px; vertical-align:middle;
    }}
    .cat-factual     {{ background:#dbeafe; color:#1d4ed8; }}
    .cat-procedural  {{ background:#dcfce7; color:#15803d; }}
    .cat-comparative {{ background:#fef9c3; color:#92400e; }}
    .cat-adversarial {{ background:#fee2e2; color:#b91c1c; }}
    code {{
      background:#f1f5f9; padding:1px 5px;
      border-radius:3px; font-size:0.85em;
    }}
    .legend {{
      display:flex; gap:1rem; font-size:0.75rem;
      color:#64748b; margin-top:0.5rem; padding:0 1.25rem 0.75rem;
    }}
    .legend-dot {{
      display:inline-block; width:10px; height:10px;
      border-radius:50%; margin-right:4px;
    }}
  </style>
</head>
<body>
  <h1>&#x1F50D; pginjection Debug</h1>
  <p class="subtitle">Explicit debug view — for retrieval validation only</p>

  <div class="query-card">
    <div class="label">Query</div>
    <div class="value">{query}{category_tag}</div>
    <div style="margin-top:6px">
      <span class="label">Filters:</span>
      <code>{_esc(filters_str)}</code>
      &nbsp;&nbsp;
      <span class="label">Chunks returned:</span>
      <strong>{len(chunks)}</strong>
    </div>
  </div>

  {answer_html}

  {metrics_html}

  <div class="table-section">
    <div class="table-header">
      <h2>&#x1F4CB; Ranked Chunks</h2>
    </div>
    <div class="legend">
      <span><span class="legend-dot" style="background:#bbf7d0"></span>score &ge; 0.85</span>
      <span><span class="legend-dot" style="background:#fef08a"></span>score 0.70&ndash;0.85</span>
      <span><span class="legend-dot" style="background:#fed7aa"></span>score &lt; 0.70</span>
    </div>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Chunk Text</th>
          <th style="text-align:right">Sim Score</th>
          <th style="text-align:right">Rerank Score</th>
          <th>Metadata</th>
        </tr>
      </thead>
      <tbody>
        {rows_html if rows_html else
         '<tr><td colspan="5" style="text-align:center;padding:2rem;color:#94a3b8">No chunks found</td></tr>'}
      </tbody>
    </table>
  </div>
</body>
</html>"""


# ── FastAPI route ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    data = _load()
    if not data:
        return HTMLResponse(
            content=(
                "<html><body style='font-family:sans-serif;padding:2rem'>"
                "<h2>&#x26A0;&#xFE0F; No debug_results.json found</h2>"
                "<p>Run <code>python main.py --debug --query 'your query'</code> first.</p>"
                "</body></html>"
            ),
            status_code=404,
        )
    html = _render_page(data)
    logger.info(f"[DEBUG_SERVER] Served debug page — {len(data.get('chunks', []))} chunks")
    return HTMLResponse(content=html)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="{time:HH:mm:ss} | {level} | {message}")
    url = f"http://{config.DEBUG_API_HOST}:{config.DEBUG_API_PORT}"
    logger.info(f"[DEBUG_SERVER] Starting at {url}")
    uvicorn.run(app, host=config.DEBUG_API_HOST, port=config.DEBUG_API_PORT, log_level="error")
