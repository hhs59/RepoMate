# RepoMate

A self-training coding copilot that learns your codebase. Point it at any Git repo — it clones, chunks, indexes, and serves a chat assistant grounded in your actual code. No code leaves your machine.

## How It Works

```
Git URL → clone → tree-sitter chunk → call graph → embed → Chroma index
                                                              ↓
                                                     RAG retrieval (BM25 + dense + reranker)
                                                              ↓
                                                     LLM generates answer with file:line citations
                                                              ↓
                                                     Optional: fine-tune on your code → hybrid mode
```

**Two modes:**
- **RAG-only (default):** Instant, free, no GPU. Retrieves relevant code + generates answers with citations.
- **Hybrid (opt-in):** Fine-tunes a model on your repo's style, combined with RAG for accuracy. Requires GPU.

## Quickstart

```bash
pip install -e ".[all]"
cp .env.example .env  # add your LLM API key

repomate init https://github.com/pallets/click
repomate serve pallets-click
```

Open http://localhost:8080 — ask questions, get answers with file:line citations.

## Docker

```bash
cp .env.example .env  # add REPOMATE_LLM_API_KEY
docker compose up
```

## CLI

| Command | What it does |
|---------|-------------|
| `repomate init <url>` | Clone, chunk, index — LLM-free, ~30s |
| `repomate serve <slug>` | Launch UI + API (RAG-only) |
| `repomate train <slug>` | Generate training data + fine-tune (needs GPU) |
| `repomate serve <slug> --hybrid` | Serve with fine-tuned adapter |

## Configuration

All settings via environment variables (`REPOMATE_` prefix) or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `REPOMATE_LLM_API_KEY` | — | Any OpenAI-compatible API key |
| `REPOMATE_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI, Together, Ollama, vLLM, etc. |
| `REPOMATE_LLM_MODEL` | `gpt-4o-mini` | Chat model for RAG generation |
| `REPOMATE_BASE_MODEL_ID` | `Qwen/Qwen2.5-1.5B-Instruct` | Model to fine-tune |
| `REPOMATE_DEVICE` | auto | `cuda`, `mps`, or `cpu` |

## Features

- **Call graph extraction** — sees caller/callee relationships, warns about blast radius when you ask to change code
- **Hybrid retrieval** — BM25 sparse + dense embeddings + cross-encoder reranking
- **Multi-level training data** — function Q&A, contextual pairs, file summaries, impact analysis, commit relabeling, AST pairs
- **Works with any LLM** — OpenAI, Together, Ollama, vLLM, any OpenAI-compatible endpoint
- **Works with any GPU** — CUDA, ROCm, Apple Silicon, or CPU fallback

## Tech Stack

Python 3.11 · FastAPI · Next.js · ChromaDB · sentence-transformers · tree-sitter · HuggingFace · TRL/PEFT · vLLM

## License

MIT
