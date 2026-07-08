# Project Report: Enterprise AI Knowledge Hub using Retrieval-Augmented Generation (RAG)

## 1. Problem Statement

Organizations accumulate large volumes of unstructured information across PDFs, Word
files, presentations, spreadsheets, manuals, and policies. Traditional keyword search
fails to understand user intent and context, leading to irrelevant results and wasted
time. This project builds an **Enterprise AI Knowledge Hub** that lets users upload
documents and ask natural-language questions, receiving accurate answers grounded in
the actual document content — with citations back to the source.

## 2. Objectives

- Build a multi-format document ingestion pipeline (PDF, DOCX, TXT, Markdown, HTML)
- Preprocess and chunk documents for efficient retrieval
- Generate vector embeddings and store them in ChromaDB
- Retrieve relevant chunks via semantic similarity search
- Generate accurate, context-aware answers using an LLM
- Display source citations with every answer
- Provide an interactive web UI for upload, querying, and document management

## 3. System Architecture

The system follows a two-stage RAG architecture:

**Stage 1 — Ingestion (offline/on-upload):**
Document → Loader → Chunking → Embedding → ChromaDB storage

**Stage 2 — Query (online, per question):**
User question → Query embedding → Similarity search in ChromaDB → Top-k relevant
chunks retrieved → Chunks + question passed to LLM → Grounded answer with citations
returned to the user.

See `diagrams/architecture_diagram.png` for the full component diagram and
`diagrams/rag_flowchart.png` for the step-by-step pipeline flowchart.

## 4. Technology Stack

| Component | Technology Used |
|---|---|
| Programming Language | Python 3.10+ |
| Orchestration Framework | LangChain |
| Vector Database | ChromaDB |
| Embedding Model | sentence-transformers (`all-MiniLM-L6-v2`), optional OpenAI embeddings |
| Large Language Model | OpenAI GPT (gpt-4o-mini) or Google Gemini (gemini-flash-latest) |
| User Interface | Streamlit |
| Document Parsing | pypdf, docx2txt, unstructured |

## 5. Implementation Details

### 5.1 Document Ingestion Pipeline
Each uploaded file is routed to a format-specific LangChain loader
(`PyPDFLoader`, `Docx2txtLoader`, `TextLoader`, `UnstructuredMarkdownLoader`,
`UnstructuredHTMLLoader`). The loader extracts raw text along with metadata
(e.g., page number for PDFs), which is preserved through the pipeline for citation
purposes.

### 5.2 Chunking Strategy
Documents are split using `RecursiveCharacterTextSplitter` with a **1000-character
chunk size** and **150-character overlap**. The splitter tries paragraph boundaries
first, then sentence, then word boundaries, which keeps chunks semantically coherent
rather than cutting text arbitrarily.

### 5.3 Embedding Generation
By default, the free, locally-run `all-MiniLM-L6-v2` sentence-transformer model is
used, requiring no API key and no per-request cost. OpenAI embeddings are available
as an optional higher-quality alternative.

### 5.4 Vector Storage & Retrieval
Chunks and their embeddings are persisted in **ChromaDB**, along with `source`
metadata identifying the originating file (and page number, where applicable).
At query time, the user's question is embedded and compared against stored vectors
using cosine similarity; the top 4 most relevant chunks are retrieved.

### 5.5 Answer Generation & Citation
Retrieved chunks are inserted into a custom prompt template that explicitly instructs
the LLM to answer **only** from the provided context and to say so plainly if the
answer isn't present — reducing hallucination. Each answer is returned alongside a
list of source documents (and snippets) used to generate it, displayed in an
expandable "Sources" panel in the UI.

### 5.6 Conversational Memory
A `ConversationBufferMemory` object retains chat history within a session, so
follow-up questions (e.g., "What about the other tier?") are correctly understood
in context of the prior exchange.

### 5.7 User Interface
Built with Streamlit for rapid, clean UI development:
- Sidebar: API key entry, embedding model choice, document upload, document list
  with delete buttons, and conversation reset
- Main panel: chat-style interface showing question/answer pairs with a
  collapsible source-citation panel under each answer

## 6. Sample Dataset

Three representative enterprise documents are included in `sample_dataset/`:
1. `hr_leave_policy.txt` — HR leave policy (annual/sick/maternity leave, WFH rules)
2. `it_security_policy.md` — IT security policy (passwords, data classification, MFA)
3. `product_faq.html` — Product FAQ (pricing tiers, integrations, compliance, support)

These allow immediate end-to-end testing without needing to source external files.

## 7. Testing & Sample Queries

| Query | Expected Behavior |
|---|---|
| "How many days of annual leave do I get?" | Answers "24 days" citing `hr_leave_policy.txt` |
| "What MFA requirements exist for remote access?" | Cites `it_security_policy.md` §6 |
| "What is the Enterprise support SLA?" | Cites `product_faq.html` |
| "What is the capital of France?" | Correctly states the documents don't contain this information |

## 8. Results & Outcome

The application successfully demonstrates the practical implementation of RAG for
enterprise use: it ingests heterogeneous document formats, retrieves semantically
relevant context (rather than relying on keyword matches), and produces grounded,
cited answers — directly addressing the reduced-search-time and improved-accessibility
goals set out in the problem statement.

## 9. Learning Outcomes Demonstrated

- End-to-end RAG architecture design and implementation using LangChain
- Multi-format document processing and metadata-preserving chunking
- Semantic search using vector embeddings and ChromaDB
- Prompt engineering to minimize hallucination and enforce citation
- Integration of LLMs (GPT / Gemini) into an interactive application
- UI/UX design for AI-powered tools using Streamlit

## 10. Future Enhancements

- Metadata filtering (e.g., restrict search to a department or document type)
- Re-ranking retrieved chunks with a cross-encoder for higher precision
- Support for scanned/image-based PDFs via OCR
- Multi-user authentication and per-user document isolation
- Deployment to a cloud environment (e.g., Streamlit Community Cloud, AWS, Azure)
