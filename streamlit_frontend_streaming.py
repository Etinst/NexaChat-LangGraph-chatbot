import streamlit as st
from frontend_helpers import ensure_history, latest_ai_text, remember, setup_page, show_history
from langchain_core.messages import HumanMessage
from langgraph_backend import chatbot

setup_page("NexaChat Groq - Streaming Style")
ensure_history()
show_history()

user_text = st.chat_input("Ask for a step-by-step explanation")
if user_text:
    remember("user", user_text)
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config={"configurable": {"thread_id": "stream-demo"}},
        )
        answer = latest_ai_text(result["messages"])
        st.write_stream([answer])
    remember("assistant", answer)
