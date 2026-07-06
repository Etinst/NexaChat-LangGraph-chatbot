from __future__ import annotations

from typing import Annotated, TypedDict

from app_settings import build_groq_llm
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

llm = build_groq_llm()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def answer_message(state: ChatState) -> dict:
    system_message = SystemMessage(
        content=(
            "You are NexaChat, a clear and friendly study assistant. "
            "Explain answers simply, ask a follow-up question when needed, "
            "and avoid pretending to know things that are not in the chat."
        )
    )
    reply = llm.invoke([system_message, *state["messages"]])
    return {"messages": [reply]}


graph = StateGraph(ChatState)
graph.add_node("answer_message", answer_message)
graph.add_edge(START, "answer_message")
graph.add_edge("answer_message", END)

chatbot = graph.compile(checkpointer=InMemorySaver())
