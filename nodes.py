import os
import math
import re
from state import AgentState
import config
from config import client, cross_encoder

# BGE models require this instruction prefix for search queries (asymmetric retrieval)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RELEVANCE_THRESHOLD = 0.80  # Calibrated for high-precision fact retrieval (~69% confidence)
RETRIEVAL_TOP_K = 7        # Increased candidate capture for multi-part & catalog queries


def sigmoid(logit: float) -> float:
    """Convert unbounded cross-encoder logits into a calibrated 0-1 probability."""
    return 1.0 / (1.0 + math.exp(-logit))


def _clean_query_keywords(text: str) -> str:
    """Extract informative keywords for dense vector retrieval on retry."""
    stop_words = {
        "what", "is", "the", "are", "how", "does", "do", "did", "of", "in", "to", "for",
        "a", "an", "and", "or", "can", "why", "might", "when", "where", "which",
        "tell", "me", "about", "describe", "explain", "it", "its", "necessary", "should"
    }
    tokens = [w for w in re.findall(r"\w+", text.lower()) if w not in stop_words]
    return " ".join(tokens) if len(tokens) >= 2 else text


def retrieve_node(state: AgentState):
    query    = state["query"]
    logs     = state.get("logs", [])
    attempts = state.get("attempts", 0) + 1

    logs.append(f"🔍 [Attempt {attempts}] Starting retrieval for query:")
    logs.append(f'   "{query}"')

    # Show which files are currently in the knowledge base (skipped in batch eval for speed)
    if os.environ.get("NO_LLM") != "1":
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
        logs.append("🔎 Searching knowledge base with BGE query instruction…")
        search_target = BGE_QUERY_PREFIX + query
        docs = config.vector_store.similarity_search(search_target, k=RETRIEVAL_TOP_K)
    else:
        logs.append("🔄 Self-correction: generating optimized search query…")
        use_llm_rewrite = (os.environ.get("NO_LLM", "0") != "1") and (client is not None)
        if use_llm_rewrite:
            try:
                rewrite_prompt = (
                    "You are an enterprise document retrieval optimizer. "
                    "Rewrite the following user query into a concise search query containing "
                    "synonyms and keywords likely found in organizational documents. "
                    "Respond with ONLY the rewritten query text and nothing else.\n\n"
                    f"Original query: {query}\n\n"
                    "Rewritten query:"
                )
                res = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=rewrite_prompt,
                )
                search_query = res.text.strip().replace('"', '')
                if not search_query:
                    search_query = _clean_query_keywords(query)
            except Exception as e:
                logs.append(f"   ⚠️ LLM rewrite fallback: {e}")
                search_query = _clean_query_keywords(query)
        else:
            search_query = _clean_query_keywords(query)

        logs.append(f'   ↳ Query reformulated as: "{search_query}"')
        search_target = BGE_QUERY_PREFIX + search_query
        docs = config.vector_store.similarity_search(search_target, k=RETRIEVAL_TOP_K)

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
    best_prob  = sigmoid(best_score)

    ranked = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)

    # Verification: single high-confidence chunk OR cumulative multi-chunk support for comparative queries
    single_pass = best_score >= RELEVANCE_THRESHOLD
    cumulative_pass = len(ranked) >= 2 and ranked[0][1] >= 0.60 and ranked[1][1] >= 0.35
    is_valid = single_pass or cumulative_pass

    logs.append(f"📊 Relevance scores for {len(docs)} chunk(s):")
    for i, (doc, score) in enumerate(ranked[:5], start=1):
        fname   = os.path.basename(doc.metadata.get("source", "Unknown"))
        page    = doc.metadata.get("page", "?")
        prob    = sigmoid(score)
        chunk_valid = score >= RELEVANCE_THRESHOLD or (cumulative_pass and score >= 0.35)
        status  = "✅" if chunk_valid else "❌"
        snippet = doc.page_content[:60].replace("\n", " ")
        logs.append(f"   [{i}] {status} Score: {score:.3f} (Conf: {prob*100:.1f}%) | {fname} p.{page} | \"{snippet}…\"")

    logs.append(f"🏆 Best score: {best_score:.4f} (Confidence: {best_prob*100:.1f}%, threshold ≥ {RELEVANCE_THRESHOLD})")

    if is_valid:
        pass_reason = "single chunk" if single_pass else "cumulative multi-chunk context"
        logs.append(f"✅ Grading PASSED via {pass_reason} — proceeding to answer generation.")
    else:
        logs.append(f"❌ Grading FAILED — best score {best_score:.4f} (Conf: {best_prob*100:.1f}%) below threshold {RELEVANCE_THRESHOLD}.")
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

    logs.append("✍️  Generating answer from verified evidence…")

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    
    # Filter out noisy chunks that did not meet the positive relevance threshold (score < 0.35)
    verified_docs = [(doc, score) for doc, score in ranked if score >= 0.35]
    top_docs = verified_docs[:3] if verified_docs else ranked[:1]

    logs.append(f"📌 Using top {len(top_docs)} verified chunk(s) as context (irrelevant chunks filtered out):")
    citations      = []
    context_blocks = []

    for rank, (doc, score) in enumerate(top_docs, start=1):
        source   = doc.metadata.get("source", "Organisation-File")
        page     = doc.metadata.get("page", 0) + 1
        filename = os.path.basename(source)
        prob     = sigmoid(score)
        citations.append(f"[{rank}] {filename} — Page/Row {page} (score: {score:.3f}, confidence: {prob*100:.1f}%)")
        context_blocks.append(f"[Source {rank}] {filename}, Page {page}:\n{doc.page_content}")
        logs.append(f"   [{rank}] {filename}  page {page}  score {score:.3f} (Conf: {prob*100:.1f}%)")

    formatted_context = "\n\n---\n\n".join(context_blocks)

    logs.append("🤖 Sending verified context to Gemini 2.5 Flash…")

    if client is None:
        final_answer = (
            "Verified context facts extracted.\n\n📋 **Audit Evidence Sources:**\n"
            + "\n".join(citations)
        )
        logs.append("✅ Context extracted (offline mode) and audit trail appended.")
        return {"answer": final_answer, "logs": logs}

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
    logs.append(f"   • No chunk scored above the relevance threshold ({RELEVANCE_THRESHOLD}).")
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