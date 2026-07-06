from __future__ import annotations

import sqlite3
from typing import Annotated, TypedDict

from app_settings import DATABASE_PATH, build_groq_llm
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

llm = build_groq_llm()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def database_chat_node(state: ChatState) -> dict:
    system_message = SystemMessage(
        content=(
            "You are NexaChat with saved conversation memory. "
            "Use the earlier messages in the thread to stay consistent."
        )
    )
    response = llm.invoke([system_message, *state["messages"]])
    return {"messages": [response]}


connection = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
checkpointer = SqliteSaver(connection)

graph = StateGraph(ChatState)
graph.add_node("database_chat_node", database_chat_node)
graph.add_edge(START, "database_chat_node")
graph.add_edge("database_chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads() -> list[str]:
    thread_ids: set[str] = set()
    for checkpoint in checkpointer.list(None):
        thread_ids.add(checkpoint.config["configurable"]["thread_id"])
    return sorted(thread_ids)
