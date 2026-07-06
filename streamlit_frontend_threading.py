import streamlit as st
from frontend_helpers import ensure_history, latest_ai_text, remember, setup_page, show_history
from langchain_core.messages import HumanMessage
from langgraph_database_backend import chatbot, retrieve_all_threads

setup_page("NexaChat Groq - Thread Manager")

saved_threads = retrieve_all_threads()
chosen = st.sidebar.selectbox("Saved threads", saved_threads or ["class-demo-thread"])
custom = st.sidebar.text_input("Use another thread id")
thread_id = custom.strip() or chosen

if st.sidebar.button("Clear visible chat"):
    st.session_state.messages = []
    st.rerun()

ensure_history()
show_history()

user_text = st.chat_input("Chat inside the selected thread")
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
