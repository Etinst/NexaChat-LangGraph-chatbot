from __future__ import annotations

import sqlite3
from typing import Annotated, TypedDict

from app_settings import DATABASE_PATH, build_groq_llm
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

llm = build_groq_llm()
web_search = DuckDuckGoSearchRun(region="us-en")


@tool
def project_note(topic: str) -> str:
    """Return a short project note that can be used during demos."""
    return (
        f"Demo note for {topic}: this Groq version keeps the LangGraph flow simple, "
        "uses environment variables for secrets, and stores chat threads in SQLite."
    )


tools = [web_search, project_note]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def mcp_style_node(state: ChatState) -> dict:
    system_message = SystemMessage(
        content=(
            "You are NexaChat in a tool-ready mode. Explain when a tool was useful, "
            "but keep the final response short and understandable."
        )
    )
    response = llm_with_tools.invoke([system_message, *state["messages"]])
    return {"messages": [response]}


connection = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
checkpointer = SqliteSaver(connection)

graph = StateGraph(ChatState)
graph.add_node("mcp_style_node", mcp_style_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "mcp_style_node")
graph.add_conditional_edges("mcp_style_node", tools_condition)
graph.add_edge("tools", "mcp_style_node")

chatbot = graph.compile(checkpointer=checkpointer)
