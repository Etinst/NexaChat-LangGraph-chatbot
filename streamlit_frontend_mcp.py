import streamlit as st
from frontend_helpers import ensure_history, latest_ai_text, remember, setup_page, show_history
from langchain_core.messages import HumanMessage
from langgraph_mcp_backend import chatbot

setup_page("NexaChat Groq - Tool Ready Demo")
st.sidebar.info("This page keeps the old MCP idea as a simple tool-ready demonstration.")

ensure_history()
show_history()

user_text = st.chat_input("Try: make a project note about LangGraph")
if user_text:
    remember("user", user_text)
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config={"configurable": {"thread_id": "mcp-demo"}},
        )
        answer = latest_ai_text(result["messages"])
        st.markdown(answer)
    remember("assistant", answer)
