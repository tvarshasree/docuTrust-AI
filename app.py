import os
import json
import shutil
import asyncio
import config

from fastapi import FastAPI, WebSocket, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from langgraph.graph import StateGraph, START, END
from state import AgentState
import nodes
from starlette.concurrency import run_in_threadpool

from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.exception_handler(Exception)
async def global_exception_handler(request: FastAPIRequest, exc: Exception):
    """Catch any unhandled exception and return JSON instead of 500 HTML."""
    msg = f"{type(exc).__name__}: {exc}"
    print(f"Unhandled error on {request.url.path}: {msg}")
    return JSONResponse(status_code=200, content={"status": "error", "message": msg})

# Build LangGraph
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", nodes.retrieve_node)
    workflow.add_node("grade", nodes.grade_node)
    workflow.add_node("generate", nodes.generate_node)
    workflow.add_node("fail", nodes.fail_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        nodes.decide_next_step,
        {"generate": "generate", "retrieve": "retrieve", "fail": "fail"},
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("fail", END)
    return workflow.compile()

rag_graph = build_graph()

# Routes
@app.get("/")
async def get_ui(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs("./documents", exist_ok=True)
    file_path = os.path.join("./documents", file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        docs = config.load_file(file_path)
    except Exception as e:
        # Delete the saved file so it doesn't linger
        try:
            os.remove(file_path)
        except Exception:
            pass
        error_msg = f"Failed to load '{file.filename}': {type(e).__name__}: {e}"
        print(f"error is:{error_msg}")
        return {"status": "error", "message": error_msg}

    if not docs:
        try:
            os.remove(file_path)
        except Exception:
            pass
        return {"status": "error", "message": f"No content extracted from '{file.filename}'. File may be empty or unsupported."}

    try:
        chunks = config.text_splitter.split_documents(docs)
        config.vector_store.add_documents(chunks)
    except Exception as e:
        error_msg = f"Failed to embed '{file.filename}': {type(e).__name__}: {e}"
        print(f"error is: {error_msg}")
        return {"status": "error", "message": error_msg}

    return {
        "status": "success",
        "message": f"✅ {file.filename} embedded — {len(chunks)} chunks added.",
    }


def _reset_store_and_docs():
    """Shared logic: wipe in-memory vector store + delete uploaded files."""
    # Reset the ChromaDB collection
    raw = config.vector_store._client
    try:
        raw.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    config.vector_store = config.Chroma(
        client=raw,
        collection_name=config.COLLECTION_NAME,
        embedding_function=config.embeddings,
    )
    print(" In-memory collection cleared and recreated.")

    # Delete uploaded files
    doc_dir = "./documents"
    if os.path.exists(doc_dir):
        for fname in os.listdir(doc_dir):
            try:
                os.remove(os.path.join(doc_dir, fname))
                print(f" Deleted {fname}")
            except Exception as e:
                print(f" Could not delete {fname}: {e}")
    os.makedirs(doc_dir, exist_ok=True)


@app.post("/on-page-load")
async def on_page_load():
    """Called by the browser on every page load/reload.
    Ensures the backend state matches the fresh UI — no stale vectors."""
    try:
        _reset_store_and_docs()
        return {"status": "ok"}
    except Exception as e:
        print(f" on_page_load error: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/clear-session")
async def clear_session():
    """Reset the vector store collection and wipe uploaded documents."""
    try:
        _reset_store_and_docs()
    except Exception as e:
        print(f" clear_session error: {e}")
        return {"status": "error", "message": f"Clear failed: {e}"}
    return {"status": "cleared", "message": "Session cleared. Knowledge base reset."}


# WebSocket chat
@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request_json = json.loads(data)
            user_query = request_json.get("query", "").strip()

            if not user_query:
                continue

            initial_state: AgentState = {
                "query": user_query,
                "retrieved_docs": [],
                "relevance_scores": [],
                "attempts": 0,
                "graded_valid": False,
                "answer": "",
                "logs": ["🚀 Initialising DocuTrust RAG pipeline..."],
            }

            await websocket.send_json({"type": "log", "message": initial_state["logs"][0]})

            def execute_graph_sync():
                return list(rag_graph.stream(initial_state))

            print(f" Running pipeline for: {user_query!r}")
            events = await run_in_threadpool(execute_graph_sync)

            for event in events:
                print(f" Graph event: {event}")
                for node_name, node_output in event.items():
                    if not isinstance(node_output, dict):
                        node_output = getattr(node_output, "__dict__", {})

                    logs = node_output.get("logs", [])
                    answer = node_output.get("answer", "")

                    if logs:
                        latest_log = logs[-1] if isinstance(logs, list) else logs
                        await websocket.send_json(
                            {"type": "log", "message": f"[{node_name.upper()}] {latest_log}"}
                        )
                    if answer:
                        await websocket.send_json({"type": "answer", "message": answer})

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)