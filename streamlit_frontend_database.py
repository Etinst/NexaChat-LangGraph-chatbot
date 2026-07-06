import streamlit as st
from frontend_helpers import ensure_history, latest_ai_text, remember, setup_page, show_history
from langchain_core.messages import HumanMessage
from langgraph_database_backend import chatbot, retrieve_all_threads

setup_page("NexaChat Groq - Saved Threads")

thread_options = retrieve_all_threads() or ["study-thread-1"]
selected_thread = st.sidebar.selectbox("Thread", thread_options)
new_thread = st.sidebar.text_input("New thread name")
thread_id = new_thread.strip() or selected_thread
st.sidebar.caption(f"Current thread: {thread_id}")

ensure_history()
show_history()

user_text = st.chat_input("Continue this saved chat")
if user_text:
    remember("user", user_text)
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        answer = latest_ai_text(result["messages"])
        st.markdown(answer)
    remember("assistant", answer)
