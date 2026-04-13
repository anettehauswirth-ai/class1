"""
FastAPI NLP Micro-API
─────────────────────
Three lightweight endpoints for health checks, text summarization,
and sentiment analysis — no ML libraries required.
"""

from __future__ import annotations

import re
import math
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
app = FastAPI(
    title="NLP Micro-API",
    description=(
        "A tiny but mighty API that summarizes text and reads the room. "
        "No GPU required — just good vibes and clever heuristics."
    ),
    version="1.0.0",
    docs_url="/",           # Swagger UI lives at the root
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────
class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text to summarize.")
    max_length: Optional[int] = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum number of sentences in the summary (1-20).",
    )

class SummarizeResponse(BaseModel):
    original_length: int
    summary_length: int
    compression_ratio: float
    summary: str

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text to analyse.")

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    explanation: str
    word_count: int
    highlights: dict


# ──────────────────────────────────────────────
# Helpers — extractive summarizer
# ──────────────────────────────────────────────
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (simple but effective)."""
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def _word_frequencies(text: str) -> dict[str, int]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
        "either", "neither", "each", "every", "all", "any", "few", "more",
        "most", "other", "some", "such", "no", "only", "own", "same", "than",
        "too", "very", "just", "because", "if", "when", "while", "where",
        "how", "what", "which", "who", "whom", "this", "that", "these",
        "those", "i", "me", "my", "we", "our", "you", "your", "he", "him",
        "his", "she", "her", "it", "its", "they", "them", "their",
    }
    freq: dict[str, int] = {}
    for w in words:
        if w not in stop and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1
    return freq


def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Extractive summarizer — picks the highest-scoring sentences."""
    sentences = _split_sentences(text)
    if len(sentences) <= max_sentences:
        return text.strip()

    freq = _word_frequencies(text)
    if not freq:
        return " ".join(sentences[:max_sentences])

    max_freq = max(freq.values())
    norm = {w: c / max_freq for w, c in freq.items()}

    scored: list[tuple[float, int, str]] = []
    for idx, sent in enumerate(sentences):
        words = re.findall(r"[a-zA-Z']+", sent.lower())
        score = sum(norm.get(w, 0) for w in words)
        # Slight position bonus: earlier sentences matter more
        position_bonus = 1.0 / (1 + idx * 0.1)
        scored.append((score * position_bonus, idx, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = sorted(scored[:max_sentences], key=lambda x: x[1])
    return " ".join(s for _, _, s in top)


# ──────────────────────────────────────────────
# Helpers — rule-based sentiment analyser
# ──────────────────────────────────────────────
_POSITIVE = {
    "good", "great", "awesome", "excellent", "amazing", "wonderful",
    "fantastic", "brilliant", "love", "loved", "loving", "happy",
    "joy", "joyful", "pleased", "glad", "delighted", "enjoy",
    "enjoyed", "enjoying", "beautiful", "perfect", "impressive",
    "outstanding", "superb", "terrific", "marvelous", "pleasant",
    "satisfied", "exciting", "excited", "thrilled", "grateful",
    "thankful", "appreciate", "appreciated", "best", "better",
    "win", "winning", "won", "success", "successful", "hope",
    "hopeful", "optimistic", "incredible", "remarkable", "fabulous",
    "like", "liked", "recommend", "recommended", "helpful",
    "easy", "elegant", "friendly", "fun", "innovative", "intuitive",
}

_NEGATIVE = {
    "bad", "terrible", "horrible", "awful", "worst", "hate", "hated",
    "hating", "angry", "anger", "sad", "sadness", "disappointed",
    "disappointing", "frustrating", "frustrated", "annoying",
    "annoyed", "ugly", "poor", "poorly", "fail", "failed", "failure",
    "boring", "bored", "dull", "pain", "painful", "unfortunately",
    "unhappy", "upset", "broken", "useless", "waste", "wrong",
    "problem", "problems", "issue", "issues", "difficult", "hard",
    "worse", "nasty", "dreadful", "miserable", "pathetic", "sucks",
    "rubbish", "trash", "lousy", "ridiculous", "stupid", "slow",
    "crash", "crashed", "bug", "bugs", "error", "errors", "lacking",
    "dislike", "disliked", "complex", "confusing", "ugly",
}

_NEGATORS = {"not", "no", "never", "neither", "nor", "barely", "hardly", "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't", "won't", "can't", "couldn't", "shouldn't", "wouldn't"}
_INTENSIFIERS = {"very", "really", "extremely", "incredibly", "absolutely", "totally", "completely", "utterly", "highly", "especially", "remarkably", "so"}


def analyze_sentiment(text: str) -> SentimentResponse:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    word_count = len(words)
    if word_count == 0:
        raise HTTPException(status_code=422, detail="Text contains no analysable words.")

    pos_score = 0.0
    neg_score = 0.0
    pos_words: list[str] = []
    neg_words: list[str] = []

    for i, w in enumerate(words):
        # Check if the previous word is a negator
        negated = i > 0 and words[i - 1] in _NEGATORS
        # Check for intensifier two words back or one word back
        intensified = (
            (i > 0 and words[i - 1] in _INTENSIFIERS)
            or (i > 1 and words[i - 2] in _INTENSIFIERS)
        )
        weight = 1.5 if intensified else 1.0

        if w in _POSITIVE:
            if negated:
                neg_score += weight
                neg_words.append(f"not {w}")
            else:
                pos_score += weight
                pos_words.append(w)
        elif w in _NEGATIVE:
            if negated:
                pos_score += weight
                pos_words.append(f"not {w}")
            else:
                neg_score += weight
                neg_words.append(w)

    total = pos_score + neg_score
    if total == 0:
        return SentimentResponse(
            sentiment="neutral",
            confidence=0.85,
            explanation="No strong sentiment signals detected — the text appears neutral or factual.",
            word_count=word_count,
            highlights={"positive": [], "negative": []},
        )

    if pos_score > neg_score:
        sentiment = "positive"
        ratio = pos_score / total
    elif neg_score > pos_score:
        sentiment = "negative"
        ratio = neg_score / total
    else:
        sentiment = "neutral"
        ratio = 0.5

    # Map ratio to a confidence between 0.5 and 0.99
    confidence = round(0.5 + 0.49 * (2 * ratio - 1) if ratio >= 0.5 else 0.5, 2)
    # Boost confidence when there are many sentiment words
    density = total / word_count
    confidence = round(min(confidence + density * 0.15, 0.99), 2)

    # Build human-readable explanation
    if sentiment == "positive":
        explanation = (
            f"The text leans positive — key signals include: {', '.join(dict.fromkeys(pos_words))}."
            + (f" Minor negative notes: {', '.join(dict.fromkeys(neg_words))}." if neg_words else "")
        )
    elif sentiment == "negative":
        explanation = (
            f"The text leans negative — key signals include: {', '.join(dict.fromkeys(neg_words))}."
            + (f" Some positive notes: {', '.join(dict.fromkeys(pos_words))}." if pos_words else "")
        )
    else:
        explanation = "Mixed signals — the text contains roughly equal positive and negative indicators."

    return SentimentResponse(
        sentiment=sentiment,
        confidence=confidence,
        explanation=explanation,
        word_count=word_count,
        highlights={
            "positive": list(dict.fromkeys(pos_words)),
            "negative": list(dict.fromkeys(neg_words)),
        },
    )


# ══════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════

@app.get("/health", tags=["Ops"])
def health_check():
    """Returns service health and current UTC timestamp."""
    return {
        "status": "healthy 🚀",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": app.version,
    }


@app.post("/summarize", response_model=SummarizeResponse, tags=["NLP"])
def summarize(req: SummarizeRequest):
    """
    Extractive text summarizer.

    Picks the most information-dense sentences from the input based on
    word-frequency scoring with a positional bias toward earlier content.
    """
    summary = summarize_text(req.text, max_sentences=req.max_length)
    orig_len = len(req.text)
    summ_len = len(summary)
    return SummarizeResponse(
        original_length=orig_len,
        summary_length=summ_len,
        compression_ratio=round(summ_len / orig_len, 2) if orig_len else 1.0,
        summary=summary,
    )


@app.post("/analyze-sentiment", response_model=SentimentResponse, tags=["NLP"])
def sentiment(req: SentimentRequest):
    """
    Rule-based sentiment analyser.

    Scans for positive / negative lexicon hits, handles negation
    (e.g. "not good" → negative) and intensifiers ("very bad" → stronger).
    Returns sentiment label, confidence, and the words that drove the decision.
    """
    return analyze_sentiment(req.text)
