# AI & LLM Engineering — 12-Week Plan

A hands-on, engineering-focused learning path to understand how AI and LLMs work and integrate them into production software.

## Structure
- `weeks/` — Week-by-week folders with goals, references, and tasks.
- `apps/` — Example services (e.g., FastAPI model server, RAG demo).
- `notebooks/` — Starter notebooks for experiments.
- `tools/` — Utilities for ingestion, evaluation, and orchestration.
- `docs/` — Design docs, ADRs, and checklists.

## Getting Started
1. **Python environment**
   ```bash
   uv venv || python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env  # Set API keys etc.
   ```

2. **Sanity check**
   ```bash
   uvicorn apps.model_server.app:app --reload
   # open http://127.0.0.1:8000/docs
   ```

3. **Pick your week**
   - Start in `weeks/week01` and work forward.
   - Use `notebooks/` for exploratory work before hardening into services in `apps/`.

## Tech Choices
- **Primary**: Python (PyTorch, HF Transformers), FastAPI, FAISS / vector DBs
- **Alt**: .NET examples via links (drop in your own C# services as needed)

## Notes
- This repo template intentionally avoids heavy pinned versions and large dependencies to make bootstrap easy.
- Some advanced steps (quantization, GPU serving) are marked optional.
