#!/usr/bin/env python3
# /// script
# dependencies = [
#   "sentence-transformers>=3.0",
#   "numpy>=1.24",
#   "anthropic>=0.45",
# ]
# ///
"""
RAG (Retrieval-Augmented Generation) — Teaching Example
========================================================

WHAT IS RAG?
------------
RAG lets a language model answer questions about documents it has never
seen during training.  The model doesn't need to be retrained — instead,
relevant passages are retrieved at query time and injected into the prompt.

This solves two core problems with "raw" LLMs:
  1. They don't know about private / recent / specialized documents.
  2. They sometimes "hallucinate" — confidently stating wrong facts.

THE PIPELINE (read the code in this order):
  ┌───────────────────────────────────────────────────────────────────────┐
  │  INDEXING   — done once, offline                                      │
  │  Documents → split into Chunks → encode each chunk as an Embedding    │
  ├───────────────────────────────────────────────────────────────────────┤
  │  RETRIEVAL  — done per query, online                                  │
  │  Query → encode → compare against chunk embeddings → Top-K chunks     │
  ├───────────────────────────────────────────────────────────────────────┤
  │  GENERATION — done per query, online                                  │
  │  Top-K chunks + Query → LLM prompt → Grounded answer                  │
  └───────────────────────────────────────────────────────────────────────┘

QUICK START:
  export ANTHROPIC_API_KEY="sk-..."
  uv run rag.py                          # interactive Q&A loop
  uv run rag.py "What is attention?"     # single question then exit

  uv automatically installs dependencies from the # /// script block above.
"""

import os
import sys
import textwrap

import numpy as np
from sentence_transformers import SentenceTransformer
import anthropic


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ── change these values to see how they affect RAG behaviour
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# A lightweight open-source model (384-dimensional vectors, ~80 MB).
# It runs on CPU and is fast enough for demos.
# Larger models like "all-mpnet-base-v2" give better quality at more cost.

CHUNK_SIZE    = 150   # maximum words per chunk
CHUNK_OVERLAP = 30    # words shared between consecutive chunks
TOP_K         = 3     # how many chunks to pass to the LLM


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE
# In a real system these would be loaded from files, a database, or a web
# crawler.  They are embedded here to keep the demo completely self-contained.
# ─────────────────────────────────────────────────────────────────────────────

DOCUMENTS = [

    # ── Document 1 ──────────────────────────────────────────────────────────
    """
    The Transformer Architecture

    The Transformer, introduced in "Attention Is All You Need" (Vaswani et al.,
    2017), is the architecture behind every modern large language model.  Unlike
    earlier RNNs it processes all tokens in parallel, making it much faster to
    train on GPUs.

    The key innovation is self-attention: every token can directly attend to
    every other token in the same sequence.  For example, in "The animal didn't
    cross the street because it was tired", attention lets the model figure out
    that "it" refers to "animal", not "street".

    Each attention head learns three weight matrices — Query (Q), Key (K), and
    Value (V) — and computes:

        Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V

    Dividing by √d_k prevents the dot products from growing too large in high
    dimensions, which would push the softmax into a region with tiny gradients.

    A Transformer layer has two sub-layers:
      • Multi-head self-attention  — captures relationships between tokens.
      • Feed-forward network       — applied independently to each position.

    Both sub-layers use residual connections and layer normalisation, which
    stabilise training of very deep stacks (hundreds of layers in GPT-4 etc.).
    """,

    # ── Document 2 ──────────────────────────────────────────────────────────
    """
    Retrieval-Augmented Generation (RAG)

    RAG is a technique that combines information retrieval with text generation.
    It was popularised by Lewis et al. (2020) at Facebook AI Research.

    The problem RAG solves: LLMs are trained on a fixed snapshot of data.
    They cannot answer questions about private documents or recent events, and
    they sometimes hallucinate plausible-sounding but wrong facts.

    RAG fixes this by retrieving relevant passages from a document store at
    query time and injecting them into the prompt.  The model is instructed to
    base its answer on those passages, which anchors it in real text.

    Classic RAG pipeline:
      1. Offline — embed all documents, store vectors in a vector database.
      2. Online  — embed the query, retrieve top-K similar chunks, append to prompt.
      3. The LLM generates an answer grounded in the retrieved context.

    Key design trade-offs:
      • Chunk size:   smaller → more precise retrieval; larger → more context per hit.
      • Top-K:        more chunks → richer context but longer (costlier) prompts.
      • Embedding model: bigger → better quality but slower indexing.
    """,

    # ── Document 3 ──────────────────────────────────────────────────────────
    """
    Embeddings and Vector Similarity

    An embedding is a dense numeric vector that represents meaning.  Two
    sentences that mean similar things will have vectors pointing in similar
    directions in the high-dimensional embedding space.

    Cosine similarity is the most common distance metric for retrieval:

        similarity(A, B) = (A · B) / (‖A‖ · ‖B‖)

    It ranges from -1 (opposite meaning) to +1 (identical meaning) and is
    independent of vector magnitude — helpful because sentence lengths vary.

    Embedding models are trained with contrastive learning: pairs of similar
    sentences are pushed close together; dissimilar sentences are pushed apart.

    Popular embedding models:
      • all-MiniLM-L6-v2       — 384 dims, fast, good quality-to-speed ratio.
      • all-mpnet-base-v2      — 768 dims, slower but more accurate.
      • text-embedding-3-small — OpenAI, strong commercial option.
      • nomic-embed-text       — open weights, competitive with commercial models.

    Vector databases (ChromaDB, FAISS, Qdrant, Pinecone) store embeddings and
    support approximate nearest-neighbour (ANN) search at scale.  For small
    corpora a plain NumPy dot-product is sufficient, as in this example.
    """,

    # ── Document 4 ──────────────────────────────────────────────────────────
    """
    Prompt Engineering

    Prompt engineering is the art of crafting the text you send to an LLM to
    reliably elicit the response you want — without changing the model weights.

    Core techniques:

    Zero-shot prompting — ask the model directly, no examples:
        "Summarise the following paragraph: …"

    Few-shot prompting — provide 2–5 input/output examples before the real
    query.  This dramatically improves performance on tasks the model has not
    been fine-tuned for.

    Chain-of-thought (CoT) — ask the model to "think step by step".  Adding
    this phrase can raise accuracy on multi-step reasoning by 10–40 pp.

    System prompts — many APIs accept a system message that sets the model's
    role, tone, and constraints before any user turn.  Example:
        "You are a helpful assistant.  Answer ONLY from the provided context.
         If the context is insufficient, say 'I don't know'."

    RAG-specific tips:
      • Tell the model explicitly where its context comes from.
      • Instruct it to admit when context is insufficient rather than guess.
      • Keep context concise — padding with irrelevant text hurts quality.
    """,

    # ── Document 5 ──────────────────────────────────────────────────────────
    """
    Neural Networks — A Refresher

    A neural network is a stack of layers.  Each layer applies a linear
    transformation (matrix multiply + bias) followed by a non-linear activation.

    Common activation functions:
      • ReLU(x)    = max(0, x)    — simple, avoids vanishing gradients, most used.
      • GELU(x)    ≈ x·Φ(x)      — smoother variant used in Transformers.
      • Sigmoid(x) = 1/(1+e⁻ˣ)  — squashes output to (0,1), used for binary tasks.

    Training uses gradient descent:
      1. Forward pass  — compute predictions.
      2. Loss          — measure how wrong the predictions are (e.g. cross-entropy).
      3. Backward pass — compute gradient of the loss w.r.t. every weight.
      4. Update        — move each weight a small step against its gradient.

    Key hyperparameters:
      • Learning rate — step size per update; too large → diverges, too small → slow.
      • Batch size    — samples per gradient step; affects stability and speed.
      • Epochs        — full passes through the training data.

    Modern LLMs are neural networks with billions of parameters trained on
    trillions of tokens across thousands of GPUs.
    """,
]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 · CHUNKING
#
# Why chunk?  Embedding models have a token limit (~256–512 tokens), and
# shorter chunks → more precise retrieval.  We use overlapping windows so
# a sentence that straddles a boundary isn't silently dropped.
# ─────────────────────────────────────────────────────────────────────────────

def chunk_document(text: str, chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split `text` into overlapping word-windows.

    Example with chunk_size=5, overlap=2:
        words  = [A, B, C, D, E, F, G]
        chunk0 = [A, B, C, D, E]
        chunk1 = [D, E, F, G]       ← D and E are shared ("overlap")
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        start += chunk_size - overlap   # slide window forward
    return chunks


def build_corpus(documents: list[str]) -> list[str]:
    """Chunk every document and return a flat list of all chunks."""
    corpus = []
    for doc in documents:
        corpus.extend(chunk_document(doc))
    return corpus


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 · INDEXING (EMBEDDING)
#
# We encode every chunk into a vector.  The result is a 2-D NumPy array
# (num_chunks × embedding_dim) that acts as our "vector database".
# ─────────────────────────────────────────────────────────────────────────────

def build_index(corpus: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    Encode every chunk in `corpus` into a normalised embedding vector.

    Returns shape (N, D) where N = number of chunks, D = embedding dimension.

    Normalising means every vector has length 1 (unit vector).  This lets us
    compute cosine similarity with a simple dot product:

        cosine_similarity(a, b) == dot(a_unit, b_unit)

    … which means retrieval is just one fast matrix multiply.
    """
    print(f"  Embedding {len(corpus)} chunks with '{EMBEDDING_MODEL}' …")
    embeddings = model.encode(corpus, show_progress_bar=False,
                              normalize_embeddings=True)
    return embeddings  # numpy array, shape (N, D)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 · RETRIEVAL
#
# At query time we encode the question and compare it against every chunk
# embedding.  The chunks with the highest cosine similarity are most relevant.
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(query: str, model: SentenceTransformer,
             index: np.ndarray, corpus: list[str],
             top_k: int = TOP_K) -> list[str]:
    """
    Find the `top_k` chunks most semantically similar to `query`.

    Steps:
      1. Encode the query into a unit vector (same space as chunk embeddings).
      2. Dot-product against every chunk embedding → similarity scores.
      3. Sort descending and return the top-k chunk texts.
    """
    query_vec = model.encode(query, normalize_embeddings=True)  # shape: (D,)
    scores    = index @ query_vec                               # shape: (N,)  ← the magic line
    top_idxs  = np.argsort(scores)[::-1][:top_k]              # highest scores first
    return [corpus[i] for i in top_idxs]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 · GENERATION
#
# We build a prompt that includes the retrieved chunks as context, then ask
# the LLM to answer the user's question based only on that context.
# ─────────────────────────────────────────────────────────────────────────────

def generate(query: str, context_chunks: list[str]) -> str:
    """
    Send the user `query` to Claude together with the retrieved `context_chunks`.

    The system prompt does two important things:
      1. Tells the model to ground its answer in the provided context only.
      2. Tells the model to say "I don't know" if the context is insufficient,
         rather than hallucinating an answer.
    """
    client  = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are a helpful teaching assistant.\n"
        "Answer the student's question using ONLY the context passages below.\n"
        "If the context does not contain enough information to answer, say so clearly "
        "and do not guess.\n\n"
        f"CONTEXT:\n{context}"
    )

    response = client.messages.create(
        model      = "claude-sonnet-4-6",
        max_tokens = 512,
        system     = system_prompt,
        messages   = [{"role": "user", "content": query}],
    )
    return response.content[0].text


# ─────────────────────────────────────────────────────────────────────────────
# MAIN  —  wire the four stages together into a working demo
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Guard: fail early with a clear message if the API key is missing
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit(
            "Error: ANTHROPIC_API_KEY is not set.\n"
            "Run:  export ANTHROPIC_API_KEY='sk-...'"
        )

    print("=" * 62)
    print("  RAG Teaching Example")
    print("=" * 62)

    # ── OFFLINE PHASE: index the knowledge base once ─────────────────────────
    print("\n[1/3] Chunking documents …")
    corpus = build_corpus(DOCUMENTS)
    print(f"      {len(DOCUMENTS)} documents  →  {len(corpus)} chunks")

    print("\n[2/3] Loading embedding model and indexing chunks …")
    model = SentenceTransformer(EMBEDDING_MODEL)
    index = build_index(corpus, model)
    print(f"      Index ready  —  shape: {index.shape}  "
          f"({index.shape[0]} chunks × {index.shape[1]} dimensions)")

    print("\n[3/3] Ready!\n")

    # ── ONLINE PHASE: answer questions ───────────────────────────────────────
    def answer(question: str) -> None:
        """Run one full retrieve-then-generate cycle and print the result."""
        print(f"Q: {question}")
        chunks      = retrieve(question, model, index, corpus)
        answer_text = generate(question, chunks)
        print("\nA:", textwrap.fill(answer_text, width=70,
                                    subsequent_indent="   "))
        print()

    # If a question was passed on the command line, answer it and exit.
    # Otherwise, drop into an interactive loop.
    if len(sys.argv) > 1:
        answer(" ".join(sys.argv[1:]))
    else:
        print("Type a question and press Enter  (or 'quit' to exit).\n")
        while True:
            try:
                q = input("Q: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in {"quit", "exit", "q"}:
                break
            answer(q)

    print("Goodbye!")


if __name__ == "__main__":
    main()
