from __future__ import annotations

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage


def setup_page(title: str) -> None:
    st.set_page_config(page_title=title, page_icon="💬", layout="centered")
    st.title(title)


def ensure_history() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def show_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def remember(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def clean_response_text(message) -> str:
    if hasattr(message, "content") and message.content:
        return str(message.content)
    if isinstance(message, ToolMessage):
        return f"Tool result: {message.content}"
    return "I received a response, but it did not contain display text."


def latest_ai_text(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and getattr(message, "content", None):
            return str(message.content)
    return clean_response_text(messages[-1])
