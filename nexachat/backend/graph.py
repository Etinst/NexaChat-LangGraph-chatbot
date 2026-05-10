"""
NexaChat - LangGraph Backend
==============================
Author: Built as portfolio project
Description:
    A stateful AI chatbot built using LangGraph's StateGraph API.
    Features multi-turn memory, tool-use (web search + calculator),
    SQLite-based persistent conversation threads, and streaming support.

Architecture:
    ┌──────────┐     ┌──────────────┐     ┌──────────┐
    │  Input   │────▶│  LLM Node    │────▶│  Tools   │
    │  State   │     │  (ChatGroq)  │     │  Node    │
    └──────────┘     └──────────────┘     └──────────┘
                            │                   │
                            ▼                   │
                       ┌─────────┐◀─────────────┘
                       │  END    │
                       └─────────┘
"""

import os
import math
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

# ──────────────────────────────────────────────
# State Definition
# ──────────────────────────────────────────────

class ChatState(TypedDict):
    """
    The state object that flows through the LangGraph nodes.
    `messages` uses the `add_messages` reducer so new messages are
    appended rather than overwriting the entire list.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    system_prompt: str


# ──────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e
    Example: '2 ** 10', 'sqrt(144)', 'sin(pi / 2)'
    """
    allowed = {
        k: v for k, v in math.__dict__.items() if not k.startswith("_")
    }
    allowed.update({"abs": abs, "round": round, "int": int, "float": float})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return f"Result: {result}"
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


@tool
def get_current_datetime() -> str:
    """Return the current date and time in ISO-8601 format."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}"


@tool
def word_counter(text: str) -> str:
    """Count the number of words, characters, and sentences in a given text."""
    words = len(text.split())
    chars = len(text)
    sentences = text.count(".") + text.count("!") + text.count("?")
    return (
        f"Words: {words} | Characters: {chars} | "
        f"Approximate sentences: {sentences}"
    )


TOOLS = [calculator, get_current_datetime, word_counter]
TOOL_MAP = {t.name: t for t in TOOLS}

# ──────────────────────────────────────────────
# LLM Setup
# ──────────────────────────────────────────────

def _build_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Please set it in your .env file."
        )
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=api_key,
    )
    return llm.bind_tools(TOOLS)


# ──────────────────────────────────────────────
# Node Functions
# ──────────────────────────────────────────────

def llm_node(state: ChatState):
    """
    Core reasoning node. Prepends the system prompt, calls the LLM,
    and returns the AI response (possibly with tool_calls).
    """
    llm = _build_llm()
    system_msg = SystemMessage(content=state.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
    response = llm.invoke([system_msg] + state["messages"])
    return {"messages": [response]}


def tools_node(state: ChatState):
    """
    Tool execution node. Runs all tool calls in the last AI message
    and returns ToolMessage results.
    """
    last_msg: AIMessage = state["messages"][-1]
    tool_results = []
    for tc in last_msg.tool_calls:
        tool_fn = TOOL_MAP.get(tc["name"])
        if tool_fn is None:
            result = f"Unknown tool: {tc['name']}"
        else:
            result = tool_fn.invoke(tc["args"])
        tool_results.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )
    return {"messages": tool_results}


def should_use_tools(state: ChatState) -> Literal["tools", "__end__"]:
    """Conditional edge: route to tools if the LLM requested any."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


# ──────────────────────────────────────────────
# Graph Assembly
# ──────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = (
    "You are NexaChat, a helpful, concise, and friendly AI assistant. "
    "You have access to tools: a calculator, a datetime checker, and a word counter. "
    "Use them when the user asks for calculations, time, or text statistics. "
    "Always be transparent when you use a tool."
)


def build_graph(db_path: str = "nexachat_memory.db"):
    """
    Compile and return the LangGraph StateGraph with SQLite checkpointing.

    Args:
        db_path: Path to the SQLite file used for conversation persistence.

    Returns:
        Compiled graph (CompiledGraph) with checkpointer attached.
    """
    builder = StateGraph(ChatState)

    builder.add_node("llm", llm_node)
    builder.add_node("tools", tools_node)

    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", should_use_tools)
    builder.add_edge("tools", "llm")  # after tools, go back to LLM for final answer

    memory = SqliteSaver.from_conn_string(db_path)
    return builder.compile(checkpointer=memory)


# ──────────────────────────────────────────────
# Public API helpers
# ──────────────────────────────────────────────

# Singleton graph instance
_graph = None

def get_graph() -> object:
    """Return (or lazily create) the compiled graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def chat(
    user_message: str,
    thread_id: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """
    Send a user message and return the assistant's reply as a plain string.

    Args:
        user_message: The text typed by the user.
        thread_id: Unique identifier for the conversation thread.
        system_prompt: Optional override for the assistant's persona.

    Returns:
        The assistant's response text.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state = {
        "messages": [HumanMessage(content=user_message)],
        "system_prompt": system_prompt,
    }
    result = graph.invoke(state, config=config)
    return result["messages"][-1].content


def stream_chat(
    user_message: str,
    thread_id: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    """
    Stream the assistant's response token-by-token.

    Yields:
        Partial content strings as they arrive from the LLM.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state = {
        "messages": [HumanMessage(content=user_message)],
        "system_prompt": system_prompt,
    }
    for chunk in graph.stream(state, config=config, stream_mode="messages"):
        if isinstance(chunk, tuple):
            msg, meta = chunk
            if hasattr(msg, "content") and msg.content:
                yield msg.content
        elif hasattr(chunk, "content") and chunk.content:
            yield chunk.content


def get_thread_history(thread_id: str) -> list[dict]:
    """
    Retrieve the full message history for a given thread.

    Returns:
        List of dicts with keys 'role' and 'content'.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state or not state.values:
        return []
    messages = state.values.get("messages", [])
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            # Skip pure tool-call messages (no visible text)
            if msg.content:
                history.append({"role": "assistant", "content": msg.content})
    return history
