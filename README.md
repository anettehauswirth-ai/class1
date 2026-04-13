# 🧠 NLP Micro-API

A lightweight FastAPI service with three endpoints for health checks, LLM-powered text summarization, and LLM-powered sentiment analysis — both powered by Claude (Anthropic). Both NLP endpoints support **custom prompts** so you can steer the model's behaviour.

---

## Setup

Both the `/summarize` and `/analyze-sentiment` endpoints require an Anthropic API key. Get one at [console.anthropic.com](https://console.anthropic.com/) and set it as an environment variable:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Endpoints

### `GET /health`
Returns service status and UTC timestamp.

```bash
curl https://your-app.onrender.com/health
```

```json
{
  "status": "healthy 🚀",
  "timestamp": "2026-04-12T18:30:00+00:00",
  "version": "1.0.0"
}
```

### `POST /summarize`
LLM-powered summarizer. Optionally pass a `custom_prompt` to control how the summary is generated. Use `{max_length}` as a placeholder for the sentence limit.

**Default behaviour:**
```bash
curl -X POST https://your-app.onrender.com/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence has transformed many industries. Healthcare uses AI for diagnostics and drug discovery. Finance relies on AI for fraud detection and algorithmic trading.",
    "max_length": 2
  }'
```

**With a custom prompt:**
```bash
curl -X POST https://your-app.onrender.com/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence has transformed many industries. Healthcare uses AI for diagnostics and drug discovery. Finance relies on AI for fraud detection and algorithmic trading.",
    "max_length": 3,
    "custom_prompt": "Summarize in {max_length} bullet points, focusing on financial impact. Return ONLY the bullet points."
  }'
```

### `POST /analyze-sentiment`
LLM-powered sentiment analysis. Optionally pass a `custom_prompt` to change the analysis perspective. The JSON response schema is enforced automatically.

**Default behaviour:**
```bash
curl -X POST https://your-app.onrender.com/analyze-sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing and very easy to use, though the documentation could be better."}'
```

**With a custom prompt:**
```bash
curl -X POST https://your-app.onrender.com/analyze-sentiment \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This product is absolutely amazing and very easy to use, though the documentation could be better.",
    "custom_prompt": "Analyse sentiment from the perspective of a product manager prioritising documentation quality."
  }'
```

**Example response:**
```json
{
  "sentiment": "negative",
  "confidence": 0.65,
  "explanation": "From a documentation-focused perspective, the core complaint — that docs could be better — signals an unmet need, despite the positive product feedback.",
  "word_count": 17,
  "highlights": {
    "positive": ["absolutely amazing", "very easy to use"],
    "negative": ["documentation could be better"]
  },
  "prompt_used": "Analyse sentiment from the perspective of a product manager prioritising documentation quality."
}
```

Both responses include a `prompt_used` field so you can see exactly which instruction was sent to the model.

---

## Run Locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) for the interactive Swagger docs.

---

## Deploy to Render

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New** → **Blueprint**.
3. Connect your GitHub repo — Render reads `render.yaml` automatically.
4. In the Render dashboard, add your `ANTHROPIC_API_KEY` under **Environment**.
5. Click **Apply** and wait for the build to finish.

Your API will be live at `https://nlp-micro-api.onrender.com`.

---

## Deploy with Docker (optional)

```bash
docker build -t nlp-micro-api .
docker run -e ANTHROPIC_API_KEY="sk-ant-..." -p 8000:8000 nlp-micro-api
```

---

## Project Structure

```
├── main.py            # FastAPI app + all endpoint logic
├── requirements.txt   # Python dependencies (fastapi, uvicorn, pydantic, httpx)
├── render.yaml        # Render deployment config
├── Dockerfile         # Container build (optional)
├── .gitignore
└── README.md
```
