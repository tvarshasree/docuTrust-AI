import os
from state import AgentState
import config
from config import client, cross_encoder


def retrieve_node(state: AgentState):
    query    = state["query"]
    logs     = state.get("logs", [])
    attempts = state.get("attempts", 0) + 1

    logs.append(f"🔍 [Attempt {attempts}] Starting retrieval for query:")
    logs.append(f'   "{query}"')

    # Show which files are currently in the knowledge base
    try:
        all_docs = config.vector_store.get()
        sources  = set()
        for meta in (all_docs.get("metadatas") or []):
            src = meta.get("source", "")
            if src:
                sources.add(os.path.basename(src))
        if sources:
            logs.append(f"📂 Knowledge base contains {len(sources)} file(s):")
            for s in sorted(sources):
                logs.append(f"   • {s}")
        else:
            logs.append("⚠️  Knowledge base is empty — please upload documents first.")
    except Exception:
        pass

    if attempts == 1:
        logs.append("🔎 Searching knowledge base…")
        docs = config.vector_store.similarity_search(query, k=5)
    else:
        search_query = f"detailed information about: {query}"
        logs.append("🔄 Self-correction: broadening search with rewritten query…")
        logs.append(f'   ↳ "{search_query}"')
        docs = config.vector_store.similarity_search(search_query, k=5)

    logs.append(f"📄 Retrieved {len(docs)} chunk(s).")

    if docs:
        seen_files = {}
        for doc in docs:
            fname = os.path.basename(doc.metadata.get("source", "Unknown"))
            page  = doc.metadata.get("page", "?")
            seen_files.setdefault(fname, set()).add(str(page))
        logs.append("📑 Chunks sourced from:")
        for fname, pages in seen_files.items():
            logs.append(f"   • {fname}  (page(s): {', '.join(sorted(pages))})")

    return {"retrieved_docs": docs, "attempts": attempts, "logs": logs}


def grade_node(state: AgentState):
    docs  = state["retrieved_docs"]
    query = state["query"]
    logs  = state["logs"]

    logs.append("⚖️  Grading chunks for relevance using Cross-Encoder…")

    if not docs:
        logs.append("⚠️  No chunks to grade — knowledge base may be empty or query too vague.")
        return {"graded_valid": False, "relevance_scores": [], "logs": logs}

    pairs  = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    best_score = float(max(scores))

    THRESHOLD = 1.5
    is_valid  = best_score >= THRESHOLD

    logs.append(f"📊 Relevance scores for {len(docs)} chunk(s):")
    ranked = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)
    for i, (doc, score) in enumerate(ranked[:5], start=1):
        fname   = os.path.basename(doc.metadata.get("source", "Unknown"))
        page    = doc.metadata.get("page", "?")
        status  = "✅" if score >= THRESHOLD else "❌"
        snippet = doc.page_content[:60].replace("\n", " ")
        logs.append(f"   [{i}] {status} Score: {score:.3f} | {fname} p.{page} | \"{snippet}…\"")

    logs.append(f"🏆 Best score: {best_score:.4f}  (threshold ≥ {THRESHOLD})")

    if is_valid:
        logs.append("✅ Grading PASSED — proceeding to answer generation.")
    else:
        logs.append(f"❌ Grading FAILED — best score {best_score:.4f} below threshold {THRESHOLD}.")
        logs.append("   Possible reasons:")
        logs.append("   • Uploaded documents may not contain information about this query.")
        logs.append("   • Query phrasing may differ too much from document language.")
        logs.append("   • Relevant content may be in a document not yet uploaded.")

    return {
        "graded_valid":     is_valid,
        "relevance_scores": scores.tolist(),
        "logs":             logs,
    }


def generate_node(state: AgentState):
    docs   = state["retrieved_docs"]
    scores = state["relevance_scores"]
    query  = state["query"]
    logs   = state["logs"]

    logs.append("✍️  Generating answer from top-ranked evidence…")

    ranked   = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = ranked[:3]

    logs.append(f"📌 Using top {len(top_docs)} chunk(s) as context:")
    citations      = []
    context_blocks = []

    for rank, (doc, score) in enumerate(top_docs, start=1):
        source   = doc.metadata.get("source", "Organisation-File")
        page     = doc.metadata.get("page", 0) + 1
        filename = os.path.basename(source)
        citations.append(f"[{rank}] {filename} — Page/Row {page}  (score: {score:.3f})")
        context_blocks.append(f"[Source {rank}] {filename}, Page {page}:\n{doc.page_content}")
        logs.append(f"   [{rank}] {filename}  page {page}  score {score:.3f}")

    formatted_context = "\n\n---\n\n".join(context_blocks)

    logs.append("🤖 Sending verified context to Gemini 2.5 Flash…")

    prompt = f"""You are DocuTrust, a secure enterprise information assistant.
Answer the user query using ONLY the verified context facts provided below.
If the context does not contain the answer, explicitly state that the verification evidence is unavailable.
Do NOT use outside knowledge. Be concise, factual, and professional.

User Query: {query}

Verified Context:
{formatted_context}

Professional Answer:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    final_answer = (
        response.text.strip()
        + "\n\n📋 **Audit Evidence Sources:**\n"
        + "\n".join(citations)
    )

    logs.append("✅ Answer generated and audit trail appended.")
    return {"answer": final_answer, "logs": logs}


def fail_node(state: AgentState):
    logs = state["logs"]
    logs.append("─" * 50)
    logs.append("⛔ PIPELINE FAILED — Self-correction limit reached.")
    logs.append("─" * 50)
    logs.append("📋 Failure summary:")
    logs.append("   • Retrieval ran twice — both attempts failed grading.")
    logs.append("   • No chunk scored above the relevance threshold (1.5).")
    logs.append("💡 Suggestions to resolve:")
    logs.append("   1. Upload the specific document that contains this information.")
    logs.append("   2. Rephrase the query using terms closer to the document's wording.")
    logs.append("   3. Check that the correct organisation's files are uploaded.")
    return {
        "answer": (
            "⚠️ DocuTrust could not find verified organisational evidence matching your query.\n\n"
            "The system searched your uploaded documents twice but no chunk scored "
            "above the relevance threshold.\n\n"
            "Suggestions:\n"
            "• Upload the document that contains this information\n"
            "• Try rephrasing the question using different wording\n"
            "• Ensure the correct organisation's files are uploaded"
        ),
        "logs": logs,
    }


def decide_next_step(state: AgentState):
    if state["graded_valid"]:
        return "generate"
    elif state["attempts"] < 2:
        state["logs"].append("🔁 Triggering self-correction loop — retrying with broader search…")
        return "retrieve"
    else:
        return "fail"