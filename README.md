# Enterprise AI Knowledge Hub (RAG)

An enterprise knowledge assistant built with **Retrieval-Augmented Generation (RAG)**.
Upload PDFs, Word files, text, Markdown, or HTML documents, then ask natural-language
questions and get accurate, context-aware answers **with source citations** — instead
of relying only on an LLM's built-in knowledge.

---

## ✨ Features

- 📥 Multi-format ingestion: PDF, DOCX, TXT, Markdown, HTML
- ✂️ Automatic preprocessing + smart chunking (overlapping, recursive splitting)
- 🧬 Embeddings via HuggingFace (free, local) or OpenAI (optional)
- 🗄️ Vector storage & semantic search using **ChromaDB**
- 🤖 Answer generation using **GPT (OpenAI)** or **Gemini (Google)**
- 📚 Every answer includes **source document + page/section citations**
- 💬 Conversational memory (follow-up questions understand context)
- 🗂️ Document management: view and delete indexed documents
- 🖥️ Clean, interactive **Streamlit** web interface

---

## 🏗️ Architecture

See `diagrams/architecture_diagram.png` and `diagrams/rag_flowchart.png`.

```
User → Streamlit UI → Document Loaders → Chunking → Embeddings → ChromaDB
                                                                     │
User Question → Embed Query → Similarity Search (ChromaDB) ─────────┘
                                        │
                              Retrieved Context + Question
                                        │
                                   LLM (GPT/Gemini)
                                        │
                        Answer + Source Citations → User
```

---

## 📁 Project Structure

```
enterprise_rag_hub/
├── app.py                      # Streamlit UI (entry point)
├── rag_pipeline.py             # Core RAG engine (ingestion, chunking, retrieval, QA)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── PROJECT_REPORT.md           # Full project report (for submission)
├── sample_dataset/             # Sample enterprise documents to test with (all 5 supported formats)
│   ├── hr_leave_policy.txt
│   ├── it_security_policy.md
│   ├── product_faq.html
│   ├── employee_expense_policy.pdf
│   └── new_hire_onboarding_handbook.docx
└── diagrams/
    ├── generate_diagrams.py
    ├── architecture_diagram.png
    └── rag_flowchart.png
```

---

## 🚀 Setup & Run (Plug and Play)

### 1. Install Python 3.10+ and create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get an LLM API key (choose one)
- **OpenAI**: https://platform.openai.com/api-keys
- **Google Gemini**: https://aistudio.google.com/app/apikey

No key is needed for embeddings — the default HuggingFace embedding model runs
locally for free. You only need a key for the LLM that generates answers.

### 4. Run the app
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501`.

### 5. Use it
1. Paste your API key into the sidebar.
2. Upload documents from `sample_dataset/` (or your own PDFs/DOCX/TXT/MD/HTML).
3. Ask a question, e.g.:
   - *"How many days of annual leave am I entitled to?"* (from `hr_leave_policy.txt`)
   - *"What is required to report a security incident?"* (from `it_security_policy.md`)
   - *"What are CloudSuite's pricing tiers?"* (from `product_faq.html`)
   - *"What is the daily meal allowance for international travel?"* (from `employee_expense_policy.pdf` — tests PDF ingestion + page-number citation)
   - *"When does health insurance coverage become active for new hires?"* (from `new_hire_onboarding_handbook.docx` — tests DOCX ingestion)
4. Expand **📚 Sources** under each answer to see exactly which document (and page,
   for PDFs) the answer was drawn from.
5. Use the 🗑️ button in the sidebar to remove a document, or "Clear conversation
   history" to start a fresh chat.

---

## 🔧 How It Works (Summary)

| Stage | Component | Detail |
|---|---|---|
| Load | `langchain_community.document_loaders` | Format-specific loaders per file type |
| Chunk | `RecursiveCharacterTextSplitter` | 1000-char chunks, 150-char overlap |
| Embed | `HuggingFaceEmbeddings` / `OpenAIEmbeddings` | `all-MiniLM-L6-v2` by default |
| Store | `Chroma` | Persisted locally to `chroma_db/` |
| Retrieve | `vectorstore.as_retriever(k=4)` | Top-4 most relevant chunks per query |
| Generate | `ConversationalRetrievalChain` | Custom prompt forces grounded, cited answers |
| Memory | `ConversationBufferMemory` | Enables natural follow-up questions |

---

## 🧩 Extending the Project

- Swap in a different vector DB (e.g., FAISS, Pinecone) by replacing the `Chroma` calls.
- Add metadata filters (e.g., filter by department or date) using ChromaDB's `where` clause.
- Add a re-ranking step (e.g., Cohere Rerank) before passing chunks to the LLM.
- Add authentication if deploying beyond local/demo use.

---

## ⚠️ Notes

- This is a submission-ready educational project; it is not hardened for production
  (no auth, no rate-limiting, keys entered client-side per session only).
- `chroma_db/` is created automatically on first run — delete it to reset the knowledge base.
