"""
Convert goldendataset.txt → ground_truth.json for metrics evaluation.

Relevance rule: a chunk is relevant for a query if its text contains
at least one of the query's expected keywords (case-insensitive).
Adversarial queries (empty keyword list) map to an empty relevant set.
"""
import sys, io, json, os
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
from dotenv import load_dotenv
import psycopg2

load_dotenv()


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
    exec(src, namespace)                        # safe: we own the file
    return namespace["GOLDEN_DATASET"]


# ── Find relevant chunk IDs by keyword presence ──────────────────────────────

def find_relevant_ids(keywords: list[str], cur, table: str, text_col: str) -> list[str]:
    if not keywords:
        return []
    conditions = " OR ".join([f"{text_col} ILIKE %s" for _ in keywords])
    sql = f"SELECT id FROM {table} WHERE {conditions}"
    params = [f"%{kw}%" for kw in keywords]
    cur.execute(sql, params)
    return [str(row[0]) for row in cur.fetchall()]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    dataset_path = Path(__file__).parent / "goldendataset.txt"
    output_path  = Path(__file__).parent / "ground_truth.json"

    items = parse_golden_dataset(str(dataset_path))
    print(f"Parsed {len(items)} golden items")

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )
    cur = conn.cursor()

    table    = os.getenv("TABLE_NAME", "sap_cash_management")
    text_col = os.getenv("TEXT_COLUMN", "content")

    ground_truth = {}
    for item in items:
        ids = find_relevant_ids(item.keywords, cur, table, text_col)
        ground_truth[item.query] = ids
        tag = f"[{item.category}]"
        print(f"  {tag:14s}  {len(ids):4d} relevant chunks  —  {item.query[:60]}")

    conn.close()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
