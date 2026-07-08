"""
app.py
------
Streamlit front-end for the Enterprise AI Knowledge Hub (RAG).

Run with:
    streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from rag_pipeline import KnowledgeHub

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


st.set_page_config(page_title="Enterprise AI Knowledge Hub", page_icon="🧠", layout="wide")

st.title("🧠 Enterprise AI Knowledge Hub")
st.caption("Retrieval-Augmented Generation (RAG) for enterprise document Q&A — with source citations.")

# ----------------------------------------------------------------------------
# Sidebar: configuration + document management
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    llm_provider = st.selectbox("LLM Provider", ["OpenAI", "Google Gemini"])
    api_key = st.text_input(f"{llm_provider} API Key", type="password", help="Your key is only used for this session, never stored.")

    embedding_choice = st.selectbox(
        "Embedding Model",
        ["HuggingFace (free, local, default)", "OpenAI (requires API key)"],
    )

    st.divider()
    st.header("📄 Document Management")

# ----------------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------------
embedding_provider = "openai" if embedding_choice.startswith("OpenAI") else "huggingface"

if "hub" not in st.session_state or st.session_state.get("embedding_provider") != embedding_provider:
    st.session_state.hub = KnowledgeHub(
        embedding_provider=embedding_provider,
        openai_api_key=api_key if embedding_provider == "openai" else None,
    )
    st.session_state.embedding_provider = embedding_provider

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

hub = st.session_state.hub

# ----------------------------------------------------------------------------
# Sidebar: upload
# ----------------------------------------------------------------------------
uploaded_files = st.sidebar.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "txt", "md", "html"],
    accept_multiple_files=True,
)

if uploaded_files:
    for f in uploaded_files:
        suffix = Path(f.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        try:
            n_chunks = hub.add_document(tmp_path, source_name=f.name)
            st.sidebar.success(f"✅ Indexed **{f.name}** ({n_chunks} chunks)")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to index {f.name}: {e}")
        finally:
            os.unlink(tmp_path)

# ----------------------------------------------------------------------------
# Sidebar: manage existing documents
# ----------------------------------------------------------------------------
sources = hub.list_sources()
if sources:
    st.sidebar.write(f"**Indexed documents ({len(sources)}):**")
    for s in sources:
        col1, col2 = st.sidebar.columns([4, 1])
        col1.write(f"📄 {s}")
        if col2.button("🗑️", key=f"del_{s}"):
            hub.delete_source(s)
            st.rerun()
else:
    st.sidebar.info("No documents indexed yet. Upload some to get started.")

if st.sidebar.button("🔄 Clear conversation history"):
    hub.reset_conversation()
    st.session_state.chat_history = []
    st.rerun()

# ----------------------------------------------------------------------------
# Main: chat interface
# ----------------------------------------------------------------------------
st.divider()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    page_info = f" (page {s['page'] + 1})" if s.get("page") is not None else ""
                    st.markdown(f"**{s['source']}**{page_info}")
                    st.caption(s["snippet"] + "…")

question = st.chat_input("Ask a question about your uploaded documents...")

if question:
    if not api_key:
        st.error("⚠️ Please enter your LLM API key in the sidebar first.")
    elif not sources:
        st.error("⚠️ Please upload at least one document before asking questions.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        try:
            if llm_provider == "OpenAI":
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
            else:
                llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0, google_api_key=api_key)

            chain = hub.build_chain(llm)

            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base and generating answer..."):
                    answer, source_docs = hub.query(chain, question)
                st.write(answer)
                if source_docs:
                    with st.expander("📚 Sources"):
                        for s in source_docs:
                            page_info = f" (page {s['page'] + 1})" if s.get("page") is not None else ""
                            st.markdown(f"**{s['source']}**{page_info}")
                            st.caption(s["snippet"] + "…")

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "sources": source_docs}
            )
        except Exception as e:
            st.error(f"Error generating response: {e}")

# ----------------------------------------------------------------------------
# Footer stats
# ----------------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.caption(f"📊 {hub.document_count()} document(s) indexed | Vector DB: ChromaDB")
