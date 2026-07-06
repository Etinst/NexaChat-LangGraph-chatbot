from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Optional, TypedDict

from app_settings import DATABASE_PATH, build_groq_llm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

llm = build_groq_llm()
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
)

THREAD_RETRIEVERS: dict[str, Any] = {}
THREAD_DOCUMENTS: dict[str, dict] = {}


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("Upload a PDF before starting document chat.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        pages = PyPDFLoader(temp_path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
        chunks = splitter.split_documents(pages)
        vector_store = FAISS.from_documents(chunks, embeddings)
        THREAD_RETRIEVERS[str(thread_id)] = vector_store.as_retriever(search_kwargs={"k": 4})
        THREAD_DOCUMENTS[str(thread_id)] = {
            "filename": filename or "uploaded.pdf",
            "pages": len(pages),
            "chunks": len(chunks),
        }
        return THREAD_DOCUMENTS[str(thread_id)]
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@tool
def document_lookup(query: str, thread_id: Optional[str] = None) -> dict:
    """Search the uploaded PDF for information related to the user's question."""
    retriever = THREAD_RETRIEVERS.get(str(thread_id))
    if retriever is None:
        return {"error": "No PDF has been uploaded for this thread yet."}
    documents = retriever.invoke(query)
    return {
        "query": query,
        "matches": [doc.page_content for doc in documents],
        "metadata": [doc.metadata for doc in documents],
        "file": THREAD_DOCUMENTS.get(str(thread_id), {}).get("filename"),
    }


web_search = DuckDuckGoSearchRun(region="us-en")
tools = [document_lookup, web_search]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def rag_chat_node(state: ChatState, config: dict | None = None) -> dict:
    thread_id = "default"
    if config:
        thread_id = config.get("configurable", {}).get("thread_id", thread_id)
    system_message = SystemMessage(
        content=(
            "You are NexaChat for PDF question answering. For document questions, "
            f"call document_lookup with thread_id='{thread_id}'. Use web_search only "
            "when the question needs outside information. If the answer is not in the PDF, say so."
        )
    )
    response = llm_with_tools.invoke([system_message, *state["messages"]], config=config)
    return {"messages": [response]}


connection = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
checkpointer = SqliteSaver(connection)

graph = StateGraph(ChatState)
graph.add_node("rag_chat_node", rag_chat_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "rag_chat_node")
graph.add_conditional_edges("rag_chat_node", tools_condition)
graph.add_edge("tools", "rag_chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads() -> list[str]:
    thread_ids: set[str] = set()
    for checkpoint in checkpointer.list(None):
        thread_ids.add(checkpoint.config["configurable"]["thread_id"])
    return sorted(thread_ids)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return THREAD_DOCUMENTS.get(str(thread_id), {})
