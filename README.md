# 🧠 NLP Micro-API

A lightweight FastAPI service with three endpoints for health checks, LLM-powered text summarization, and LLM-powered sentiment analysis — both powered by Claude (Anthropic).

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
LLM-powered summarizer — sends text to Claude and returns an abstractive summary.

```bash
curl -X POST https://your-app.onrender.com/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence has transformed many industries. Healthcare uses AI for diagnostics and drug discovery. Finance relies on AI for fraud detection and algorithmic trading. Education is being reshaped by personalized learning systems. Transportation is evolving with self-driving vehicles. The future of AI holds both great promise and significant challenges.",
    "max_length": 2
  }'
```

```json
{
  "original_length": 370,
  "summary_length": 105,
  "compression_ratio": 0.28,
  "summary": "AI is transforming industries including healthcare, finance, education, and transportation. Its future holds both significant promise and notable challenges."
}
```

### `POST /analyze-sentiment`
LLM-powered sentiment analysis — sends text to Claude and returns a structured assessment with confidence score, explanation, and highlighted words.

```bash
curl -X POST https://your-app.onrender.com/analyze-sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing and very easy to use, though the documentation could be better."}'
```

```json
{
  "sentiment": "positive",
  "confidence": 0.82,
  "explanation": "The text is overwhelmingly positive, praising the product as 'amazing' and 'easy to use,' with only a minor critique about documentation.",
  "word_count": 17,
  "highlights": {
    "positive": ["absolutely amazing", "very easy to use"],
    "negative": ["documentation could be better"]
  }
}
```

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
