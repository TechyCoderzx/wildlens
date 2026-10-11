# WildLens
**Discover the wild. Understand what you see.**

WildLens is a local-first nature-identification web app for exploring plants, birds, and insects from photographs. The initial scaffold uses React + Vite, FastAPI, and Ollama. Model identification is experimental and must not be treated as verified scientific identification.

## Current status
- [x] Frontend and backend starter structure
- [x] Health endpoint and Ollama connectivity/model endpoint
- [x] Image upload API with validation and structured response shape
- [x] Minimal React upload-and-analyze screen
- [ ] Validate Gemma 3n E4B image inference on your machine
- [ ] IndexedDB field journal and offline app-shell caching
- [ ] Sentry Agent Tracing
- [ ] Optional ElevenLabs narration

## Prerequisites
- Python 3.10+
- Node.js 20+
- Ollama installed and running
- A vision-capable Ollama model pulled locally

## 1. Start Ollama and verify the model
In PowerShell:
```powershell
ollama --version
ollama list
```
Check available disk space before downloading a model. After choosing and pulling a model, verify it appears in `ollama list`. This project defaults to `gemma3n:e4b`; confirm the exact model tag available in your Ollama library before relying on it.

## 2. Start the backend
From the repository root, in PowerShell:
```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Check `http://127.0.0.1:8000/health` and `http://127.0.0.1:8000/docs`.

## 3. Start the frontend
Open a second terminal:
```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```
Open the URL Vite prints (usually `http://localhost:5173`).

## API
- `GET /health` — backend status
- `GET /api/v1/vision/status` — Ollama availability and configured model
- `POST /api/v1/vision/analyze` — multipart image upload (`image` field), returns a structured, cautious identification

The analysis endpoint expects Ollama's `/api/chat` vision support. If the model tag is unavailable or the model cannot process images, it returns a useful error instead of fabricating a result.

## Local-first and privacy notes
- Uploaded images are sent from the browser to the local FastAPI service and then to the local Ollama service.
- No image is intentionally saved by the backend.
- The initial scaffold does not implement the journal, service worker, tracing, or narration yet.
- A local model reduces dependence on per-request hosted inference fees, but uses local compute and electricity.
- Do not expose the Ollama port or this development API to an untrusted network.

## Project layout
See `docs/ARCHITECTURE.md` for the component responsibilities and next steps.
