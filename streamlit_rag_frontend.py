import streamlit as st
from frontend_helpers import ensure_history, latest_ai_text, remember, setup_page, show_history
from langchain_core.messages import HumanMessage
from langraph_rag_backend import chatbot, ingest_pdf, thread_document_metadata, thread_has_document

setup_page("NexaChat Groq - PDF RAG")

thread_id = st.sidebar.text_input("Thread", value="pdf-thread-1")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file and st.sidebar.button("Index PDF"):
    summary = ingest_pdf(uploaded_file.getvalue(), thread_id=thread_id, filename=uploaded_file.name)
    st.sidebar.success(f"Indexed {summary['pages']} pages into {summary['chunks']} chunks")

if thread_has_document(thread_id):
    details = thread_document_metadata(thread_id)
    st.sidebar.info(f"Current PDF: {details.get('filename')} ({details.get('chunks')} chunks)")
else:
    st.sidebar.warning("Upload and index a PDF before asking document questions.")

ensure_history()
show_history()

user_text = st.chat_input("Ask a question about the uploaded PDF")
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
