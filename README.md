# NexaChat Groq LangGraph Chatbot

**Made by:** Darshan Nitin Barhate

NexaChat is a Streamlit chatbot project that uses LangGraph to control the chat flow and Groq to generate answers. The project contains small separate apps so a beginner can learn one feature at a time: basic chat, saved chat threads, tool use, streaming-style chat, and PDF question answering.

This version was improved to make the code easier to run, easier to understand, and safer for a normal student project.

## Main Features

- Basic chatbot using LangGraph.
- Groq LLM support through `langchain-groq`.
- Saved conversations using SQLite checkpoints.
- Tool-based chatbot with web search, calculator, and optional stock quotes.
- PDF question answering using FAISS and HuggingFace embeddings.
- Multiple Streamlit frontends for different chatbot modes.
- Shared settings file so API keys and model names are not repeated everywhere.

## Improvements Made

- Added `app_settings.py` for common settings like `GROQ_API_KEY`, `GROQ_MODEL`, database path, and LLM creation.
- Added `.env.example` so users know which environment variables are needed.
- Kept the LLM part Groq-based.
- Used local HuggingFace embeddings for PDF search.
- Added `frontend_helpers.py` to avoid repeating the same Streamlit message code in every file.
- Rewrote the Streamlit pages with clearer names, sidebars, and simple chat flow.
- Added better project documentation using this README and a LaTeX report.
- Validated the Python files for syntax errors.

## Project Files

| File | Purpose |
| --- | --- |
| `app_settings.py` | Reads environment variables and creates the Groq chat model. |
| `frontend_helpers.py` | Common Streamlit helper functions for page setup and chat history. |
| `langgraph_backend.py` | Simple LangGraph chatbot backend. |
| `langgraph_database_backend.py` | Chatbot backend with saved thread memory using SQLite. |
| `langgraph_tool_backend.py` | Chatbot backend with tools such as search, calculator, and stock quote. |
| `langgraph_mcp_backend.py` | Tool-ready demo backend written in a simple style. |
| `langraph_rag_backend.py` | PDF RAG backend using FAISS and HuggingFace embeddings. |
| `streamlit_frontend.py` | Basic chat app. |
| `streamlit_frontend_database.py` | App for saved conversation threads. |
| `streamlit_frontend_tool.py` | App for tool-based chat. |
| `streamlit_frontend_mcp.py` | App for the MCP-style demo. |
| `streamlit_frontend_streaming.py` | App showing a simple streaming-style interface. |
| `streamlit_frontend_threading.py` | App focused on thread selection. |
| `streamlit_rag_frontend.py` | PDF upload and document question-answering app. |
| `NexaChat_Groq_Improvement_Report.tex` | LaTeX project report. |
| `NexaChat_Groq_Student_Improved.ipynb` | Small notebook summary of the improved project. |

## Setup

### 1. Install Python Packages

Use Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

### 2. Create `.env`

Copy `.env.example` to `.env` and fill in your values.

```text
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
ALPHAVANTAGE_API_KEY=optional_stock_api_key_here
NEXACHAT_DB=nexachat_threads.db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

`ALPHAVANTAGE_API_KEY` is optional. It is only needed for the stock quote tool.

## How to Run

Run any one Streamlit app at a time.

```bash
streamlit run streamlit_frontend.py
```

Other useful commands:

```bash
streamlit run streamlit_frontend_database.py
streamlit run streamlit_frontend_tool.py
streamlit run streamlit_rag_frontend.py
```

## Simple Working Explanation

1. The user types a message in Streamlit.
2. Streamlit sends the message to a LangGraph backend.
3. LangGraph decides the next step in the chat flow.
4. Groq generates the chatbot answer.
5. If tools are enabled, LangGraph can call a tool and then continue the answer.
6. In the PDF app, the uploaded PDF is split into small chunks and stored in FAISS.
7. When the user asks about the PDF, the app searches the useful chunks and gives them to the chatbot.

## PDF RAG Notes

The PDF app uses two parts:

- `HuggingFaceEmbeddings` converts PDF text chunks into vectors.
- `FAISS` stores and searches those vectors.

This keeps the PDF search part separate from the Groq chat model and makes the project easier to understand.

## Troubleshooting

- If the app says the Groq key is missing, check that `.env` contains `GROQ_API_KEY`.
- If PDF chat is slow the first time, it may be downloading the embedding model.
- If stock quotes do not work, add `ALPHAVANTAGE_API_KEY` or use the other tools.
- If a tool call fails, try a Groq model that supports tool calling.

## Student Learning Points

This project helped improve understanding of:

- Streamlit web apps
- LangGraph workflow design
- Environment variables and API keys
- SQLite-based chat history
- Tool calling in chatbots
- Basic RAG using PDF text, embeddings, and vector search

