import os
from state import AgentState
import config
from config import client, cross_encoder


def retrieve_node(state: AgentState):
    query = state["query"]
    logs = state.get("logs", [])
    attempts = state.get("attempts", 0) + 1

    logs.append(f"[Attempt {attempts}] Retrieving semantic chunks for query...")

    # On retry, broaden the search angle
    search_query = query if attempts == 1 else f"detailed information about: {query}"

    docs = config.vector_store.similarity_search(search_query, k=5)
    return {"retrieved_docs": docs, "attempts": attempts, "logs": logs}


def grade_node(state: AgentState):
    docs = state["retrieved_docs"]
    query = state["query"]
    logs = state["logs"]

    logs.append("Grading retrieved chunks for contextual relevance via Cross-Encoder...")

    if not docs:
        logs.append("⚠️ No documents retrieved — grading skipped.")
        return {"graded_valid": False, "relevance_scores": [], "logs": logs}

    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    best_score = float(max(scores))

    logs.append(f"Best Cross-Encoder relevance score: {best_score:.4f}")

    THRESHOLD = 1.5
    is_valid = best_score >= THRESHOLD

    logs.append("✅ Documents PASSED relevance grading." if is_valid else "❌ Documents FAILED relevance grading.")

    return {
        "graded_valid": is_valid,
        "relevance_scores": scores.tolist(),
        "logs": logs,
    }


def generate_node(state: AgentState):
    docs = state["retrieved_docs"]
    scores = state["relevance_scores"]
    query = state["query"]
    logs = state["logs"]

    logs.append("Compiling verified answer with strict source citations...")

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = ranked[:3]

    citations = []
    context_blocks = []

    for rank, (doc, score) in enumerate(top_docs, start=1):
        source = doc.metadata.get("source", "Organisation-File")
        # page is 0-indexed in LangChain PDF loader
        page = doc.metadata.get("page", 0) + 1
        filename = os.path.basename(source)
        citations.append(f"[{rank}] {filename} — Page/Row {page}  (score: {score:.3f})")
        context_blocks.append(f"[Source {rank}] {filename}, Page {page}:\n{doc.page_content}")

    formatted_context = "\n\n---\n\n".join(context_blocks)

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

    logs.append("✅ Final answer synthesized and audit trail appended.")
    return {"answer": final_answer, "logs": logs}


def fail_node(state: AgentState):
    logs = state["logs"]
    logs.append("⛔ Self-correction limit reached. Pipeline terminated.")
    return {
        "answer": (
            "⚠️ DocuTrust could not find verified organisational evidence matching your query.\n"
            "Please upload the relevant document or rephrase your question."
        ),
        "logs": logs,
    }


# ── Router ─────────────────────────────────────────────────────────────────────
def decide_next_step(state: AgentState):
    if state["graded_valid"]:
        return "generate"
    elif state["attempts"] < 2:
        state["logs"].append("Relevance threshold not met — triggering self-correction loop...")
        return "retrieve"
    else:
        return "fail"