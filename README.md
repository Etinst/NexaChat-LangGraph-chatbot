# LangGraph Chatbot Suite

A modular, production-ready conversational AI system built with **LangGraph**, **LangChain**, and **Streamlit**. The project is structured as a collection of progressively advanced chatbot backends paired with matching frontends — from a simple single-thread chatbot all the way to a full RAG (Retrieval-Augmented Generation) pipeline with tool use and PDF ingestion.

---

## Project Overview

This suite demonstrates how to build stateful, multi-turn AI chatbots using LangGraph's graph-based orchestration. Each backend/frontend pair introduces a new capability on top of the last, making it a great reference for learning or extending LangGraph-powered applications.

---

## File Structure

```
.
├── Backends
│   ├── langgraph_backend.py           # Basic in-memory chatbot
│   ├── langgraph_database_backend.py  # SQLite-persisted chatbot
│   ├── langgraph_tool_backend.py      # Chatbot with tool use (search, calculator, stocks)
│   ├── langgraph_mcp_backend.py       # Async chatbot with MCP server integration
│   └── langraph_rag_backend.py        # Full RAG chatbot with PDF ingestion
│
├── Frontends
│   ├── streamlit_frontend.py          # Basic single-thread UI
│   ├── streamlit_frontend_streaming.py  # Streaming token output
│   ├── streamlit_frontend_database.py   # Multi-thread UI with conversation history
│   ├── streamlit_frontend_threading.py  # Tool chatbot with thread sidebar
│   ├── streamlit_frontend_tool.py       # Tool chatbot with live tool status
│   ├── streamlit_frontend_mcp.py        # Async MCP chatbot UI
│   └── streamlit_rag_frontend.py        # PDF chatbot UI with upload & RAG
│
└── requirements.txt
```

---

## Backend Descriptions

### `langgraph_backend.py` — Basic Chatbot
The simplest setup. Uses `InMemorySaver` as the checkpointer, so conversation state is lost when the server restarts. Good starting point for understanding LangGraph's `StateGraph` and message passing.

### `langgraph_database_backend.py` — Persistent Chatbot
Same as the basic backend but uses `SqliteSaver` to persist conversation history to a local `chatbot.db` SQLite database. Includes a `retrieve_all_threads()` helper to list past conversation threads.

### `langgraph_tool_backend.py` — Tool-Enabled Chatbot
Extends the database backend with three tools bound to the LLM:
- **DuckDuckGo Search** — live web search
- **Calculator** — arithmetic operations (add, subtract, multiply, divide)
- **Stock Price Lookup** — fetches real-time prices via Alpha Vantage API

Uses `ToolNode` and `tools_condition` to handle tool call routing inside the graph.

### `langgraph_mcp_backend.py` — Async MCP Chatbot
An async-first backend that runs on a dedicated event loop in a background thread. Connects to external MCP (Model Context Protocol) servers via `MultiServerMCPClient`, enabling integration with custom tool servers (e.g. a math server and an expense tracker). Also includes the search and stock price tools. Uses `AsyncSqliteSaver` for persistence.

### `langraph_rag_backend.py` — RAG Chatbot with PDF Support
The most feature-complete backend. Adds PDF ingestion using:
- **PyPDFLoader** to parse documents
- **FAISS** vector store for similarity search
- **OpenAI Embeddings** (`text-embedding-3-small`) for chunking and indexing

A `rag_tool` is registered with the LLM so it can retrieve relevant chunks from uploaded PDFs on a per-thread basis. Also includes search, stock price, and calculator tools. Thread-level document metadata is tracked in memory.

---

## Frontend Descriptions

### `streamlit_frontend.py` — Basic UI
Minimal single-thread chat interface. Fixed `thread-1` config. No sidebar or thread management.

### `streamlit_frontend_streaming.py` — Streaming UI
Same as the basic frontend but uses `chatbot.stream()` with `stream_mode="messages"` to stream tokens to the UI in real time.

### `streamlit_frontend_database.py` — Multi-Thread UI
Full conversation management UI with:
- New Chat button to start fresh threads
- Sidebar listing all past conversations (loaded from SQLite)
- Click any thread to reload its history
- Streaming responses

### `streamlit_frontend_threading.py` — Tool Chatbot UI (Simple)
Pairs with `langgraph_tool_backend.py`. Adds a thread sidebar and filters the stream to only display `AIMessage` tokens (skipping tool call noise).

### `streamlit_frontend_tool.py` — Tool Chatbot UI (with Status)
Same as above but adds a live `st.status()` indicator that shows which tool is currently running and marks it complete when done.

### `streamlit_frontend_mcp.py` — Async MCP UI
Pairs with `langgraph_mcp_backend.py`. Runs the async graph stream in the backend event loop via a `queue.Queue` bridge, yielding only AI tokens to the Streamlit `write_stream` interface. Shows live tool status for MCP tool calls.

### `streamlit_rag_frontend.py` — PDF Chatbot UI
The most complete frontend:
- PDF file uploader in the sidebar with live indexing status
- Shows indexed document metadata (filename, pages, chunks)
- Full thread management with past conversation loading
- Live tool status indicators
- Displays document context info below each response

---

## Setup & Installation

### 1. Clone the repository and install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

For the stock price tool, the Alpha Vantage API key is hardcoded in the backend files. Replace it with your own key from [alphavantage.co](https://www.alphavantage.co).

### 3. Run a frontend

```bash
# Basic chatbot
streamlit run streamlit_frontend.py

# Tool chatbot
streamlit run streamlit_frontend_tool.py

# RAG/PDF chatbot
streamlit run streamlit_rag_frontend.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT (via `langchain-openai`) |
| Orchestration | LangGraph (`StateGraph`, `ToolNode`) |
| Persistence | SQLite (`SqliteSaver` / `AsyncSqliteSaver`) |
| Vector Store | FAISS |
| Embeddings | OpenAI `text-embedding-3-small` |
| Web Search | DuckDuckGo (`langchain-community`) |
| MCP Integration | `langchain-mcp-adapters` |
| Frontend | Streamlit |

---

## Notes

- All backends use **thread-based memory** via LangGraph's checkpointing system. Each `thread_id` represents an independent conversation.
- The RAG backend stores retrievers **in-memory per thread**, so PDF indexes are lost on server restart. For production, persist the FAISS index to disk.
- The MCP backend requires running MCP servers at the configured addresses. Update the server URLs/paths in `langgraph_mcp_backend.py` to match your setup.
