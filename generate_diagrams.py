"""
generate_diagrams.py
---------------------
Generates the system architecture diagram and RAG pipeline flowchart
as PNG images for the project report / submission.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

COLOR_UI = "#4C6EF5"
COLOR_APP = "#12B886"
COLOR_STORE = "#F59F00"
COLOR_LLM = "#E64980"
COLOR_DOC = "#868E96"


def box(ax, xy, w, h, text, color, fontsize=10, text_color="white"):
    rect = patches.FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor="black", facecolor=color, alpha=0.9
    )
    ax.add_patch(rect)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
             color=text_color, weight="bold", wrap=True)


def arrow(ax, start, end, text=None, color="black"):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                         linewidth=1.4, color=color)
    ax.add_patch(a)
    if text:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.12, text, ha="center", fontsize=8, style="italic")


# ---------------------------------------------------------------------------
# 1. SYSTEM ARCHITECTURE DIAGRAM
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_title("Enterprise AI Knowledge Hub — System Architecture", fontsize=15, weight="bold", pad=15)

# User / UI layer
box(ax, (0.5, 6.3), 2.6, 1.1, "User\n(Web Browser)", COLOR_DOC)
box(ax, (4.0, 6.3), 3.2, 1.1, "Streamlit UI\n(Upload / Chat / Manage)", COLOR_UI)

# Document ingestion pipeline
box(ax, (0.5, 4.3), 2.6, 1.1, "Document Loaders\nPDF / DOCX / TXT / MD / HTML", COLOR_DOC)
box(ax, (3.4, 4.3), 2.6, 1.1, "Preprocessing &\nChunking (LangChain)", COLOR_APP)
box(ax, (6.3, 4.3), 2.6, 1.1, "Embedding Model\n(HuggingFace / OpenAI)", COLOR_APP)
box(ax, (9.2, 4.3), 3.2, 1.1, "ChromaDB\n(Vector Store)", COLOR_STORE)

# Query pipeline
box(ax, (4.0, 2.2), 3.2, 1.1, "RAG Orchestrator\n(rag_pipeline.py)", COLOR_APP)
box(ax, (7.6, 2.2), 2.8, 1.1, "Semantic\nSimilarity Search", COLOR_STORE)
box(ax, (4.0, 0.3), 3.2, 1.1, "LLM\n(GPT / Gemini)", COLOR_LLM)
box(ax, (7.6, 0.3), 2.8, 1.1, "Answer + Source\nCitations", COLOR_LLM)

# Arrows: upload/ingestion flow
arrow(ax, (1.8, 6.3), (1.8, 5.4), "upload docs")
arrow(ax, (3.1, 4.85), (3.4, 4.85))
arrow(ax, (6.0, 4.85), (6.3, 4.85))
arrow(ax, (8.9, 4.85), (9.2, 4.85), "store vectors")

# Arrows: query flow
arrow(ax, (5.6, 6.3), (5.6, 5.4))
arrow(ax, (5.6, 4.3), (5.6, 3.3), "ask question")
arrow(ax, (7.2, 2.75), (7.6, 2.75), "retrieve top-k")
arrow(ax, (9.0, 2.2), (9.0, 1.4))
arrow(ax, (7.2, 3.5), (10.5, 4.3), "vector lookup", color="gray")
arrow(ax, (5.6, 2.2), (5.6, 1.4), "context + question")
arrow(ax, (7.2, 0.85), (7.6, 0.85), "generate answer")
arrow(ax, (8.6, 3.4), (5.6, 0.9), "", color="none")

plt.tight_layout()
plt.savefig("architecture_diagram.png", dpi=180, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# 2. RAG PIPELINE FLOWCHART
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 12))
ax.set_xlim(0, 6.5)
ax.set_ylim(0, 12)
ax.axis("off")
ax.set_title("RAG Pipeline — Flowchart", fontsize=15, weight="bold", pad=15)

steps = [
    ("Upload Document\n(PDF/DOCX/TXT/MD/HTML)", COLOR_DOC),
    ("Load & Extract Text\n(LangChain Document Loaders)", COLOR_DOC),
    ("Split into Chunks\n(RecursiveCharacterTextSplitter)", COLOR_APP),
    ("Generate Embeddings\n(Sentence-Transformers / OpenAI)", COLOR_APP),
    ("Store Vectors in ChromaDB\n(with source metadata)", COLOR_STORE),
    ("User Asks a Question", COLOR_UI),
    ("Embed the Query", COLOR_APP),
    ("Semantic Similarity Search\n(Top-k retrieval from ChromaDB)", COLOR_STORE),
    ("Build Prompt\n(retrieved context + question + history)", COLOR_APP),
    ("LLM Generates Answer\n(GPT / Gemini)", COLOR_LLM),
    ("Return Answer + Source Citations\nto User Interface", COLOR_LLM),
]

y = 11.3
box_h = 0.85
gap = 0.28
xs = 1.0
w = 4.5

centers = []
for label, color in steps:
    box(ax, (xs, y - box_h), w, box_h, label, color, fontsize=9.5)
    centers.append((xs + w / 2, y - box_h))
    y -= (box_h + gap)

for i in range(len(centers) - 1):
    x0, y0 = centers[i]
    x1, y1 = centers[i + 1]
    arrow(ax, (x0, y0), (x1, y1 + box_h))

plt.tight_layout()
plt.savefig("rag_flowchart.png", dpi=180, bbox_inches="tight")
plt.close()

print("Diagrams generated: architecture_diagram.png, rag_flowchart.png")
