"""
rag_pipeline.py
----------------
Core Retrieval-Augmented Generation (RAG) engine for the Enterprise AI Knowledge Hub.

Deliberately avoids the legacy `langchain.chains` / `langchain.memory` /
`langchain.prompts` modules, which have moved or changed across LangChain
versions and caused install headaches. Instead it talks to the LLM and
vector store directly, which works reliably across LangChain 0.2.x - 0.3.x+.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "enterprise_docs"

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
}

SYSTEM_PROMPT = """You are an Enterprise AI Knowledge Assistant.
Answer the question using ONLY the information in the context below.
If the context does not contain the answer, say clearly that the uploaded
documents do not contain enough information — do NOT make anything up.
Be concise, accurate, and professional."""


# --------------------------------------------------------------------------
# Document loading
# --------------------------------------------------------------------------
def load_document(file_path: str):
    """Load a single document using the loader appropriate to its extension."""
    ext = Path(file_path).suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if not loader_cls:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {list(LOADER_MAP.keys())}"
        )
    loader = loader_cls(file_path)
    return loader.load()


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def chunk_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Split documents into semantically meaningful, overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
def get_embedding_function(provider: str = "huggingface", openai_api_key: Optional[str] = None):
    """
    Returns an embedding function.
    - 'huggingface' -> free, runs locally, no API key needed (default).
    - 'openai'      -> higher quality, requires an OpenAI API key.
    """
    if provider == "openai":
        if not openai_api_key:
            raise ValueError("OpenAI API key required for OpenAI embeddings.")
        return OpenAIEmbeddings(openai_api_key=openai_api_key)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# --------------------------------------------------------------------------
# Knowledge Hub — main orchestrator class
# --------------------------------------------------------------------------
class KnowledgeHub:
    def __init__(
        self,
        persist_dir: str = PERSIST_DIR,
        embedding_provider: str = "huggingface",
        openai_api_key: Optional[str] = None,
    ):
        self.persist_dir = persist_dir
        self.embedding_fn = get_embedding_function(embedding_provider, openai_api_key)
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_fn,
            collection_name=COLLECTION_NAME,
        )
        self.chat_history: List[Tuple[str, str]] = []

    # ---------------- Ingestion ----------------
    def add_document(self, file_path: str, source_name: Optional[str] = None) -> int:
        """Ingest one document end-to-end: load -> chunk -> embed -> store."""
        docs = load_document(file_path)
        display_name = source_name or Path(file_path).name
        for d in docs:
            d.metadata["source"] = display_name

        chunks = chunk_documents(docs)
        if chunks:
            self.vectorstore.add_documents(chunks)
            self.vectorstore.persist()
        return len(chunks)

    # ---------------- Document management ----------------
    def list_sources(self) -> List[str]:
        """Return the distinct set of source document names currently indexed."""
        data = self.vectorstore.get()
        sources = set()
        for meta in data.get("metadatas", []) or []:
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)

    def delete_source(self, source_name: str):
        """Remove all chunks belonging to a given source document."""
        self.vectorstore._collection.delete(where={"source": source_name})
        self.vectorstore.persist()

    def document_count(self) -> int:
        return len(self.list_sources())

    # ---------------- Retrieval + Generation ----------------
    def build_chain(self, llm):
        """
        Stores the LLM on the hub for use in query(). Kept as a separate method
        (rather than folding into __init__) so app.py can swap providers/models
        without recreating the whole KnowledgeHub / vector store.
        """
        self.llm = llm
        return self

    def _format_history(self, max_turns: int = 3) -> str:
        if not self.chat_history:
            return ""
        recent = self.chat_history[-max_turns:]
        lines = []
        for q, a in recent:
            lines.append(f"Previous Question: {q}\nPrevious Answer: {a}")
        return "\n\n".join(lines)

    @staticmethod
    def _extract_text(content) -> str:
        """
        Normalize an LLM response into plain text.
        Newer models (e.g. Gemini 3.x 'thinking' models) can return `content`
        as a list of structured blocks (e.g. [{"type": "text", "text": "..."},
        {"type": "thinking", "extras": {...}}]) instead of a plain string.
        This pulls out only the actual answer text and ignores thinking/
        signature metadata blocks.
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text" and "text" in block:
                        parts.append(block["text"])
                    elif "text" in block:
                        parts.append(block["text"])
            if parts:
                return "\n".join(parts).strip()

        return str(content)

    def query(self, chain, question: str):
        """
        Run a question through the RAG pipeline and return (answer, source_citations).
        `chain` is unused directly (kept for interface compatibility with app.py)
        since build_chain() already stashed the llm on self.
        """
        docs = self.vectorstore.similarity_search(question, k=4)

        context = "\n\n---\n\n".join(d.page_content for d in docs)
        history_block = self._format_history()

        user_prompt = f"""{SYSTEM_PROMPT}

{("Conversation so far:\n" + history_block) if history_block else ""}

Context:
{context}

Question: {question}

Answer:"""

        response = self.llm.invoke(user_prompt)
        raw_content = response.content if hasattr(response, "content") else response
        answer = self._extract_text(raw_content)

        sources = []
        seen = set()
        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            key = (src, page)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "source": src,
                    "page": page,
                    "snippet": doc.page_content[:250].strip(),
                }
            )

        self.chat_history.append((question, answer))
        return answer, sources

    def reset_conversation(self):
        self.chat_history = []
