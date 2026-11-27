# Contributing to NLP Pipeline — Open-Domain QA System

Thank you for your interest in contributing!

## Development Setup

1. Clone the repository and follow the setup steps in [README.md](README.md).
2. Download model artifacts:
   ```bash
   python scripts/download_models.py
   ```
3. Install Python dependencies (`backend/requirements.txt`) and frontend dependencies (`pnpm install`).

## Running Checks Before Opening a PR

- **Backend tests**:
  ```bash
  python -m unittest discover -s tests -v
  ```
- **Frontend production build**:
  ```bash
  pnpm build
  ```

## Guidelines

- Keep backend changes cleanly typed with Pydantic request validation.
- Do not commit large binaries (`*.pt`, `*.safetensors`, `*.pkl`, `*.npz`, `glove*.txt`), `node_modules/`, or `dist/`.
- Keep API response schemas in `backend/main.py` and `src/app/api/pipeline.ts` synchronized.
