# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Script

Dependencies are declared inline via PEP 723 and managed by `uv`:

```bash
# Interactive Q&A loop (OpenAI)
export OPENAI_API_KEY="sk-..."
uv run rag.py

# Single question then exit
uv run rag.py "What is attention?"

# LiteLLM proxy (any OpenAI-compatible backend)
export OPENAI_API_KEY="anything"
export LLM_BASE_URL="http://localhost:4000"
export LLM_MODEL="ollama/llama3"
uv run rag.py
```

There is no test suite, linter config, or build step.

## Architecture

`rag.py` is a single-file, self-contained RAG demo with four sequential stages:

1. **Chunking** (`chunk_document`, `build_corpus`) — splits each document in `DOCUMENTS` into overlapping word-windows. `CHUNK_SIZE` and `CHUNK_OVERLAP` control window size and stride.

2. **Indexing** (`build_index`) — encodes every chunk with `sentence-transformers` (`all-MiniLM-L6-v2`) into unit-normalised vectors, producing a `(N_chunks × 384)` NumPy matrix that serves as the in-memory vector store.

3. **Retrieval** (`retrieve`) — encodes the query with the same model, then computes cosine similarity via a single matrix multiply (`index @ query_vec`), returning the top-K chunks by score.

4. **Generation** (`generate`) — injects the top-K chunks into a system prompt and calls an OpenAI-compatible chat endpoint. The system prompt instructs the model to answer only from the provided context.

The knowledge base (`DOCUMENTS` list in `rag.py`) is hardcoded — five passages covering Transformers, RAG, embeddings, prompt engineering, and neural networks. To use a different corpus, replace or extend `DOCUMENTS`.

`rag_pipeline.html` is a standalone teaching diagram of the pipeline — no build tooling required, open directly in a browser.

## Key Configuration Constants

| Constant | Default | Effect |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Swap to `all-mpnet-base-v2` for higher quality |
| `CHUNK_SIZE` | `150` words | Smaller → more precise retrieval |
| `CHUNK_OVERLAP` | `30` words | Prevents sentences from being silently split |
| `TOP_K` | `3` | Chunks passed to the LLM; more = richer context but longer prompts |
| `LLM_MODEL` | `gpt-4o-mini` | Override via `LLM_MODEL` env var |
| `LLM_BASE_URL` | `""` (OpenAI) | Set to a LiteLLM proxy URL to use local models |
