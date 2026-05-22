# Ray RAG Backend

This backend mirrors the provided n8n workflow and exposes a POST endpoint at `/ray-rag-model`.

## Setup

1. Copy `.env.example` to `.env` and fill in the required values.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Endpoint

`POST /ray-rag-model`

Accepts flexible input fields (`chatInput`, `body.chatInput`, `query.chatInput`, `text`, `message`) and returns:

```json
{
  "answer": "...",
  "source": "...",
  "confidence": "...",
  "sources": []
}
```
