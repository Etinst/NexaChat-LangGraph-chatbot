# NexaChat 🤖

> **A production-quality AI chatbot built with LangGraph, Groq LLaMA-3, and Streamlit.**  
> Features stateful multi-turn memory, tool-use, multi-thread conversations, and a beautiful dark UI.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **LangGraph StateGraph** | Explicit node/edge architecture with conditional tool routing |
| 💾 **Persistent Memory** | SQLite-backed checkpointing — conversations survive restarts |
| 🔧 **Built-in Tools** | Calculator · Current DateTime · Word Counter |
| 🎭 **Persona Presets** | Switch assistant personalities per conversation |
| 💬 **Multi-Thread UI** | Create/switch conversations in the sidebar |
| ⚡ **Streaming** | Token-by-token responses with live typing indicator |
| 📥 **Export** | Download any conversation as Markdown |
| 🌙 **Dark Glassmorphism UI** | Animated bubbles, avatars, and responsive layout |

---

## 🚀 Quick Start

```bash
# 1. Clone / unzip the project
cd nexachat

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY (free at https://console.groq.com)

# 5. Run the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

---

## 🗂 Project Structure

```
nexachat/
├── app.py                    ← Streamlit entry point (run this)
├── requirements.txt
├── .env.example              ← Copy to .env, add your key
├── .gitignore
├── .streamlit/
│   └── config.toml           ← Dark theme config
├── backend/
│   ├── __init__.py
│   └── graph.py              ← LangGraph StateGraph, nodes, tools
└── utils/
    ├── __init__.py
    └── helpers.py            ← Thread helpers, persona presets
```

---

## 🏗 Architecture

```
User Input
    │
    ▼
┌───────────────────────────────────────┐
│          LangGraph StateGraph          │
│                                       │
│  START ──▶ [ LLM Node ] ──┬──▶ END   │
│                            │          │
│                    (has tool_calls?)  │
│                            │          │
│                            ▼          │
│                    [ Tools Node ]     │
│                            │          │
│                            └──▶ LLM  │
└───────────────────────────────────────┘
    │
    ▼
SQLite Checkpointer (persistent memory per thread_id)
```

**Nodes:**
- **LLM Node** — Calls Groq LLaMA-3.3-70B with the system prompt + message history. May emit tool_calls.
- **Tools Node** — Executes requested tools and returns ToolMessage results.
- **Conditional Edge** — Routes back to LLM after tool execution, or ends if no tools needed.

---

## 🔧 Tools

| Tool | Description |
|---|---|
| `calculator` | Safely evaluates math expressions using Python's `math` module |
| `get_current_datetime` | Returns the current UTC date/time |
| `word_counter` | Counts words, characters, and sentences in a text |

Add your own tool with the `@tool` decorator in `backend/graph.py`.

---

## 🎭 Persona Presets

Switch the assistant's personality from the sidebar:

- **NexaChat (Default)** — General assistant with tool access
- **Code Mentor** — Senior engineer who explains the "why"
- **Writing Coach** — Journalism-trained editor
- **Data Analyst** — Python data expert
- **Explain Like I'm 5** — Complex ideas in simple language

---

## 📦 Dependencies

| Package | Role |
|---|---|
| `langgraph` | StateGraph orchestration & SQLite checkpointing |
| `langchain-groq` | Groq LLM integration |
| `langchain-core` | Message types, tool decorators |
| `streamlit` | Web UI framework |
| `python-dotenv` | `.env` loading |

---

## 📄 License

MIT License — free for personal and commercial use.
