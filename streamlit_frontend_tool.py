import streamlit as st
from frontend_helpers import ensure_history, latest_ai_text, remember, setup_page, show_history
from langchain_core.messages import HumanMessage
from langgraph_tool_backend import chatbot, retrieve_all_threads

setup_page("NexaChat Groq - Tools")
st.sidebar.write("Tools enabled: web search, calculator, stock quote")
thread_id = st.sidebar.text_input("Thread", value=(retrieve_all_threads() or ["tool-thread-1"])[0])

ensure_history()
show_history()

user_text = st.chat_input("Ask a question that may need a tool")
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
