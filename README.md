# 🧠 NLP Micro-API

A lightweight FastAPI service with three endpoints for health checks, extractive text summarization, and rule-based sentiment analysis — no ML libraries or GPU required.

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
Extractive summarizer — picks the most information-dense sentences.

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
  "summary_length": 119,
  "compression_ratio": 0.32,
  "summary": "Artificial intelligence has transformed many industries. Healthcare uses AI for diagnostics and drug discovery."
}
```

### `POST /analyze-sentiment`
Rule-based sentiment analysis with negation and intensifier handling.

```bash
curl -X POST https://your-app.onrender.com/analyze-sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing and very easy to use, though the documentation could be better."}'
```

```json
{
  "sentiment": "positive",
  "confidence": 0.87,
  "explanation": "The text leans positive — key signals include: amazing, easy. Minor negative notes: better.",
  "word_count": 17,
  "highlights": {
    "positive": ["amazing", "easy"],
    "negative": ["better"]
  }
}
```

---

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) for the interactive Swagger docs.

---

## Deploy to Render

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New** → **Blueprint**.
3. Connect your GitHub repo — Render reads `render.yaml` automatically.
4. Click **Apply** and wait for the build to finish.

Your API will be live at `https://nlp-micro-api.onrender.com`.

---

## Deploy with Docker (optional)

```bash
docker build -t nlp-micro-api .
docker run -p 8000:8000 nlp-micro-api
```

---

## Project Structure

```
├── main.py            # FastAPI app + all endpoint logic
├── requirements.txt   # Python dependencies
├── render.yaml        # Render deployment config
├── Dockerfile         # Container build (optional)
├── .gitignore
└── README.md
```
