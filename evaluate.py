"""
DocuTrust RAG Evaluation Suite  — End-to-End (tests real nodes.py pipeline)
=============================================================================
Usage:
    python evaluate.py               # full end-to-end with Gemini (slow, uses API quota)
    python evaluate.py --no-llm      # retrieval-only, no Gemini calls (fast, offline)
    python evaluate.py --llm-rpm 2   # limit to 2 Gemini calls/min (for very low quotas)

What is tested:
  - config.load_file()           your file loaders
  - config.text_splitter         your chunker
  - config.embeddings            your embedding model
  - nodes.retrieve_node()        your retrieval logic + retry broadening
  - nodes.grade_node()           your cross-encoder grader + threshold (1.5)
  - nodes.decide_next_step()     your retry / fail routing
  - nodes.generate_node()        your Gemini answer generation
  - nodes.fail_node()            your fallback message

Metrics:
  - Answer Correctness  : GT answer words present in Gemini's final answer
  - Groundedness        : cross-encoder score (normalised 0-1) — from grade_node
  - Hallucination Rate  : inverse of Answer Correctness
  - Pipeline Pass Rate  : % queries that passed grade_node (not routed to fail)
  - Retry Rate          : % queries that needed a second retrieval attempt
  - Fail Rate           : % queries that hit fail_node

Results saved to: evaluation_results/
  - full_results.csv
  - summary_by_org.csv
  - summary_by_filetype.csv
  - summary_by_org_filetype.csv
  - overall_metrics.csv
  - evaluation_report.txt
"""

import os, sys, ast, json, time, warnings, gc, argparse
import pandas as pd
import chromadb
from io import StringIO
from langchain_chroma import Chroma

# ── CLI args (parsed early so USE_LLM is available at module level) ───────────
_ap = argparse.ArgumentParser(description="DocuTrust RAG Evaluation")
_ap.add_argument(
    "--no-llm", action="store_true",
    help="Skip Gemini generate_node. Answer Correctness measured against "
         "retrieved context instead of the generated answer. Runs offline, no API quota used."
)
_ap.add_argument(
    "--llm-rpm", type=int, default=4,
    help="Gemini requests per minute allowed (default 4, free tier limit is 5). "
         "Only used when --no-llm is NOT set."
)
CLI_ARGS, _ = _ap.parse_known_args()
USE_LLM  = not CLI_ARGS.no_llm
LLM_RPM  = CLI_ARGS.llm_rpm
# Seconds to wait between Gemini calls to stay under rate limit
LLM_DELAY = 60.0 / LLM_RPM   # e.g. 4 rpm → 15s gap

warnings.filterwarnings("ignore")

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import nodes
from state import AgentState

# ── Settings ───────────────────────────────────────────────────────────────────
DATASET_CSV = "test_data/Dataset categories - queries_01122025.csv"
RESULTS_DIR = "evaluation_results"

ORGS = [
    "Aventro Motors",
    "Cendara University",
    "Cloudway 24",
    "Velvera Technologies",
    "ZX Bank",
]

FILE_TYPES = [".pdf", ".docx", ".html", ".md", ".pptx"]

# Thresholds for PASS/FAIL verdict on each metric
THRESHOLDS = {
    "Answer Correctness": 0.70,
    "Groundedness":       0.60,
    "Hallucination Rate": 0.20,
}

METRIC_COLS = list(THRESHOLDS.keys())

_log_buffer = StringIO()

def log(msg=""):
    print(msg, flush=True)
    _log_buffer.write(msg + "\n")


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_facts(raw) -> list[dict]:
    if pd.isna(raw) or not str(raw).strip():
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            r = parser(str(raw))
            return r if isinstance(r, list) else [r]
        except Exception:
            pass
    return []


def context_contains(context: str, answer_text: str, threshold: float = 0.35) -> bool:
    ctx_words = set(context.lower().split())
    ans_words = answer_text.lower().split()
    if not ans_words:
        return False
    hits = sum(1 for w in ans_words if w in ctx_words)
    return (hits / len(ans_words)) >= threshold


def metric_fmt(v, metric):
    th = THRESHOLDS[metric]
    ok = v <= th if metric == "Hallucination Rate" else v >= th
    return f"{v:.3f} {'✅' if ok else '❌'}"


def passes(row) -> bool:
    return (
        row["Answer Correctness"] >= THRESHOLDS["Answer Correctness"] and
        row["Groundedness"]       >= THRESHOLDS["Groundedness"]       and
        row["Hallucination Rate"] <= THRESHOLDS["Hallucination Rate"]
    )


def print_table(rows: list[dict], title: str):
    if not rows:
        return
    log(f"\n{'═'*100}")
    log(f"  {title}")
    log(f"{'═'*100}")
    try:
        from tabulate import tabulate
        log(tabulate(rows, headers="keys", tablefmt="grid",
                     stralign="left", floatfmt=".3f"))
    except ImportError:
        keys = list(rows[0].keys())
        log("  " + " | ".join(f"{k:<22}" for k in keys))
        log("  " + "─" * (24 * len(keys)))
        for r in rows:
            log("  " + " | ".join(f"{str(r.get(k,'')):<22}" for k in keys))


def save_csv(df: pd.DataFrame, filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False)
    log(f"  Saved → {path}")


# ── Build a fresh vector store for one org+filetype ───────────────────────────

def build_store(org: str, ext: str) -> tuple:
    """
    Load all files of the given extension for the org into a fresh
    EphemeralClient Chroma store using config.load_file() and
    config.text_splitter — exactly what app.py /upload does.
    """
    folder = f"test_data/{org}"
    raw_client = chromadb.EphemeralClient()
    store = Chroma(
        client=raw_client,
        collection_name="eval",
        embedding_function=config.embeddings,
    )
    docs = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if not fname.lower().endswith(ext):
                continue
            fpath = os.path.join(root, fname)
            try:
                docs.extend(config.load_file(fpath))
            except Exception as e:
                log(f"       Skip {fname}: {e}")
    if not docs:
        return store, raw_client, 0
    chunks = config.text_splitter.split_documents(docs)
    store.add_documents(chunks)
    return store, raw_client, len(chunks)


# ── Run one query through the REAL nodes.py pipeline ─────────────────────────

_last_llm_call = 0.0   # epoch time of last Gemini call (module-level for rate limiting)

def run_query_pipeline(store: Chroma, query: str, gt_facts: list[dict]) -> dict:
    """
    Injects `store` into config.vector_store so that nodes.retrieve_node()
    uses the per-org/per-filetype store, then runs the full pipeline:
        retrieve_node → grade_node → decide_next_step → generate_node / fail_node

    If USE_LLM=False (--no-llm flag), generate_node is skipped and Answer
    Correctness is measured against the retrieved context instead.

    Rate limiting: enforces LLM_DELAY seconds between Gemini calls to avoid
    429 RESOURCE_EXHAUSTED on free-tier accounts.
    """
    global _last_llm_call

    original_store = config.vector_store
    config.vector_store = store

    try:
        state: AgentState = {
            "query":            query,
            "retrieved_docs":   [],
            "relevance_scores": [],
            "attempts":         0,
            "graded_valid":     False,
            "answer":           "",
            "logs":             [],
        }

        # ── retrieve → grade (with one retry) ─────────────────────────────
        state = {**state, **nodes.retrieve_node(state)}
        state = {**state, **nodes.grade_node(state)}
        attempts = state["attempts"]

        next_step = nodes.decide_next_step(state)
        if next_step == "retrieve":
            state = {**state, **nodes.retrieve_node(state)}
            state = {**state, **nodes.grade_node(state)}
            attempts  = state["attempts"]
            next_step = nodes.decide_next_step(state)

        pipeline_passed = state["graded_valid"]
        best_ce_score   = float(max(state["relevance_scores"])) \
                          if state["relevance_scores"] else 0.0

        # ── retrieved context (used for Answer Correctness in --no-llm mode)
        retrieved_context = "\n\n".join(
            d.page_content for d in state.get("retrieved_docs", [])
        )

        # ── generate or fail ───────────────────────────────────────────────
        if next_step == "generate":
            if USE_LLM:
                # Rate-limit: wait if we called Gemini too recently
                gap = time.time() - _last_llm_call
                if gap < LLM_DELAY:
                    wait = LLM_DELAY - gap
                    log(f"      ⏳ rate-limit pause {wait:.1f}s …")
                    time.sleep(wait)

                # Retry once on 429
                for attempt_llm in range(2):
                    try:
                        state = {**state, **nodes.generate_node(state)}
                        _last_llm_call = time.time()
                        break
                    except Exception as e:
                        err = str(e)
                        if "429" in err or "RESOURCE_EXHAUSTED" in err:
                            retry_wait = 60
                            # Parse suggested retry delay from error if present
                            import re as _re
                            m = _re.search(r"retry in (\d+)", err, _re.IGNORECASE)
                            if m:
                                retry_wait = int(m.group(1)) + 5
                            log(f"      ⚠️  Gemini 429 — waiting {retry_wait}s then retrying…")
                            time.sleep(retry_wait)
                            if attempt_llm == 1:
                                # Still failing — fall back gracefully
                                log("      ❌ Gemini still rate-limited; using context as answer.")
                                state["answer"] = retrieved_context
                        else:
                            log(f"      ❌ Gemini error: {e}")
                            state["answer"] = retrieved_context
                            break
            else:
                # --no-llm: use retrieved context as the proxy answer
                state["answer"] = retrieved_context
        else:
            state = {**state, **nodes.fail_node(state)}

        final_answer = state.get("answer", "")

    finally:
        config.vector_store = original_store

    # ── metrics ────────────────────────────────────────────────────────────
    gt_texts = [f["text"] for f in gt_facts if "text" in f]

    ans_cor = 0.0
    if gt_texts and final_answer:
        correct = sum(1 for gt in gt_texts if context_contains(final_answer, gt))
        ans_cor = correct / len(gt_texts)

    groundedness  = max(0.0, min(1.0, (best_ce_score + 5) / 15))
    hallucination = 1.0 - ans_cor

    return {
        "Answer Correctness": round(ans_cor,       4),
        "Groundedness":       round(groundedness,  4),
        "Hallucination Rate": round(hallucination, 4),
        "Pipeline Passed":    pipeline_passed,
        "Attempts":           attempts,
        "Answer":             final_answer[:300],
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log("=" * 100)
    log("   DocuTrust RAG Reliability Evaluation  (End-to-End — real nodes.py pipeline)")
    mode_str = "RETRIEVAL-ONLY (--no-llm)" if not USE_LLM else f"FULL END-TO-END with Gemini (rate limit: {LLM_RPM} rpm)"
    log(f"   Mode   : {mode_str}")
    log(f"   Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 100)

    if not os.path.exists(DATASET_CSV):
        log(f" CSV not found: {DATASET_CSV}")
        sys.exit(1)

    df_csv = pd.read_csv(DATASET_CSV)
    df_csv.columns = [c.strip() for c in df_csv.columns]
    log(f" {len(df_csv)} rows loaded | Columns: {list(df_csv.columns)}\n")

    col_org   = next((c for c in df_csv.columns if "enterprise" in c.lower()), None) or \
                next((c for c in df_csv.columns if "org" in c.lower()), None)
    col_qtype = next((c for c in df_csv.columns if "type" in c.lower()), None)
    col_query = next((c for c in df_csv.columns if c.lower() == "query"), None) or \
                next((c for c in df_csv.columns if "query" in c.lower()), None)
    col_facts = next((c for c in df_csv.columns if "supporting" in c.lower() or
                      "fact" in c.lower()), None)

    if not all([col_org, col_query, col_facts]):
        log(f" Could not detect required columns. Found: {list(df_csv.columns)}")
        sys.exit(1)

    log(f" Columns → org:'{col_org}'  qtype:'{col_qtype}'  "
        f"query:'{col_query}'  facts:'{col_facts}'\n")

    all_results = []

    for org in ORGS:
        org_folder = f"test_data/{org}"
        if not os.path.exists(org_folder):
            log(f"  Skipping {org} — folder not found")
            continue

        org_queries = df_csv[df_csv[col_org].str.strip() == org]
        if org_queries.empty:
            log(f"  No CSV rows for '{org}'")
            continue

        log(f"\n  {org}  ({len(org_queries)} queries)")

        for ext in FILE_TYPES:
            ext_label = ext.lstrip(".")

            log(f"   Ingesting {ext_label.upper()} files…")
            store, raw_client, n_chunks = build_store(org, ext)
            if n_chunks == 0:
                log(f"      no {ext_label} files found — skipping")
                continue
            log(f"      {n_chunks} chunks loaded ✅")

            log(f"    Running {len(org_queries)} queries through nodes.py pipeline…")
            for _, row in org_queries.iterrows():
                query      = str(row[col_query]).strip()
                query_type = str(row[col_qtype]).strip() if col_qtype else "—"
                gt_facts   = parse_facts(row[col_facts])

                metrics = run_query_pipeline(store, query, gt_facts)
                passed  = passes(metrics)

                all_results.append({
                    "Organisation":       org,
                    "File Type":          ext_label.upper(),
                    "Query Type":         query_type,
                    "Query":              query,
                    "Answer Correctness": metrics["Answer Correctness"],
                    "Groundedness":       metrics["Groundedness"],
                    "Hallucination Rate": metrics["Hallucination Rate"],
                    "Pipeline Passed":    "✅" if metrics["Pipeline Passed"] else "❌",
                    "Attempts":           metrics["Attempts"],
                    "Pass":               "✅" if passed else "❌",
                })

            log(f"      done ✅")

            # Destroy store before next batch
            try:
                raw_client.delete_collection("eval")
            except Exception:
                pass
            del store
            gc.collect()
            log(f"     {ext_label.upper()} store cleared\n")

    # ── Build full dataframe ───────────────────────────────────────────────────
    if not all_results:
        log("\n No results collected.")
        return

    full_df = pd.DataFrame(all_results)
    full_df["_passed"] = full_df.apply(passes, axis=1)

    # ── TABLE 1: Full per-query results ───────────────────────────────────────
    t1_rows = []
    for _, r in full_df.iterrows():
        t1_rows.append({
            "Organisation": r["Organisation"],
            "File Type":    r["File Type"],
            "Query Type":   r["Query Type"],
            "Query":        r["Query"][:65] + ("…" if len(r["Query"]) > 65 else ""),
            "Ans Correct":  metric_fmt(r["Answer Correctness"], "Answer Correctness"),
            "Grounded":     metric_fmt(r["Groundedness"],       "Groundedness"),
            "Hallucinate":  metric_fmt(r["Hallucination Rate"], "Hallucination Rate"),
            "Pipe Pass":    r["Pipeline Passed"],
            "Attempts":     r["Attempts"],
            "Pass":         r["Pass"],
        })
    print_table(t1_rows, "TABLE 1 — FULL QUERY-LEVEL RESULTS")

    # ── TABLE 2: By Organisation ──────────────────────────────────────────────
    org_rows = []
    for org in full_df["Organisation"].unique():
        sub = full_df[full_df["Organisation"] == org]
        pipe_pass = (sub["Pipeline Passed"] == "✅").mean()
        retry     = (sub["Attempts"] == 2).mean()
        fail      = (sub["Pipeline Passed"] == "❌").mean()
        org_rows.append({
            "Organisation":   org,
            "Total Queries":  len(sub),
            "Ans Correct":    metric_fmt(sub["Answer Correctness"].mean(), "Answer Correctness"),
            "Grounded":       metric_fmt(sub["Groundedness"].mean(),       "Groundedness"),
            "Hallucinate":    metric_fmt(sub["Hallucination Rate"].mean(), "Hallucination Rate"),
            "Pipe Pass%":     f"{pipe_pass*100:.1f}%",
            "Retry Rate%":    f"{retry*100:.1f}%",
            "Fail Rate%":     f"{fail*100:.1f}%",
            "Pass Rate":      f"{sub['_passed'].mean()*100:.1f}%",
        })
    print_table(org_rows, "TABLE 2 — SUMMARY BY ORGANISATION")

    # ── TABLE 3: By File Type ─────────────────────────────────────────────────
    ft_rows = []
    for ft in [e.lstrip(".").upper() for e in FILE_TYPES]:
        sub = full_df[full_df["File Type"] == ft]
        if sub.empty:
            continue
        pipe_pass = (sub["Pipeline Passed"] == "✅").mean()
        retry     = (sub["Attempts"] == 2).mean()
        fail      = (sub["Pipeline Passed"] == "❌").mean()
        ft_rows.append({
            "File Type":     ft,
            "Total Queries": len(sub),
            "Ans Correct":   metric_fmt(sub["Answer Correctness"].mean(), "Answer Correctness"),
            "Grounded":      metric_fmt(sub["Groundedness"].mean(),       "Groundedness"),
            "Hallucinate":   metric_fmt(sub["Hallucination Rate"].mean(), "Hallucination Rate"),
            "Pipe Pass%":    f"{pipe_pass*100:.1f}%",
            "Retry Rate%":   f"{retry*100:.1f}%",
            "Fail Rate%":    f"{fail*100:.1f}%",
            "Pass Rate":     f"{sub['_passed'].mean()*100:.1f}%",
        })
    print_table(ft_rows, "TABLE 3 — SUMMARY BY FILE TYPE")

    # ── TABLE 4: Organisation × File Type ─────────────────────────────────────
    combo_rows = []
    for org in full_df["Organisation"].unique():
        for ft in [e.lstrip(".").upper() for e in FILE_TYPES]:
            sub = full_df[(full_df["Organisation"] == org) & (full_df["File Type"] == ft)]
            if sub.empty:
                continue
            pipe_pass = (sub["Pipeline Passed"] == "✅").mean()
            combo_rows.append({
                "Organisation": org,
                "File Type":    ft,
                "Queries":      len(sub),
                "Ans Correct":  metric_fmt(sub["Answer Correctness"].mean(), "Answer Correctness"),
                "Grounded":     metric_fmt(sub["Groundedness"].mean(),       "Groundedness"),
                "Hallucinate":  metric_fmt(sub["Hallucination Rate"].mean(), "Hallucination Rate"),
                "Pipe Pass%":   f"{pipe_pass*100:.1f}%",
                "Pass Rate":    f"{sub['_passed'].mean()*100:.1f}%",
            })
    print_table(combo_rows, "TABLE 4 — ORGANISATION × FILE TYPE")

    # ── TABLE 5: Overall system performance ───────────────────────────────────
    overall_rows = []
    for m in METRIC_COLS:
        v  = full_df[m].mean()
        th = THRESHOLDS[m]
        direction = "≤" if m == "Hallucination Rate" else "≥"
        ok = v <= th if m == "Hallucination Rate" else v >= th
        overall_rows.append({
            "Metric":    m,
            "Score":     round(v, 4),
            "Threshold": f"{direction}{th}",
            "Status":    "✅ PASS" if ok else "❌ FAIL",
        })

    # Pipeline-specific headline metrics
    pipe_pass_rate = (full_df["Pipeline Passed"] == "✅").mean()
    retry_rate     = (full_df["Attempts"] == 2).mean()
    fail_rate      = (full_df["Pipeline Passed"] == "❌").mean()
    op             = full_df["_passed"].mean()

    for label, val, thr, direction in [
        ("Pipeline Pass Rate", pipe_pass_rate, 0.80, "≥"),
        ("Retry Rate",         retry_rate,     0.30, "≤"),
        ("Fail Rate",          fail_rate,       0.10, "≤"),
        ("Overall Pass Rate",  op,              0.50, "≥"),
    ]:
        ok = val <= float(thr) if direction == "≤" else val >= float(thr)
        overall_rows.append({
            "Metric":    label,
            "Score":     round(val, 4),
            "Threshold": f"{direction}{thr}",
            "Status":    "✅ PASS" if ok else "❌ FAIL",
        })

    print_table(overall_rows, "TABLE 5 — OVERALL SYSTEM PERFORMANCE")

    # ── Save all CSVs ─────────────────────────────────────────────────────────
    log(f"\n{'═'*100}")
    log("  SAVING RESULTS")
    log(f"{'═'*100}")

    save_csv(full_df.drop(columns=["_passed"]), "full_results.csv")
    save_csv(
        full_df.groupby("Organisation")[METRIC_COLS].mean().round(4).reset_index(),
        "summary_by_org.csv"
    )
    save_csv(
        full_df.groupby("File Type")[METRIC_COLS].mean().round(4).reset_index(),
        "summary_by_filetype.csv"
    )
    save_csv(
        full_df.groupby(["Organisation", "File Type"])[METRIC_COLS].mean().round(4).reset_index(),
        "summary_by_org_filetype.csv"
    )
    save_csv(
        pd.DataFrame([{
            "Metric": r["Metric"], "Score": r["Score"],
            "Threshold": r["Threshold"], "Status": r["Status"]
        } for r in overall_rows]),
        "overall_metrics.csv"
    )

    report_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(_log_buffer.getvalue())
    log(f"   Saved → {report_path}")

    log(f"\n{'═'*100}")
    log(f"   Evaluation complete — all results in ./{RESULTS_DIR}/")
    log(f"{'═'*100}\n")


if __name__ == "__main__":
    t0 = time.time()
    main()
    log(f"  Total time: {time.time()-t0:.1f}s")