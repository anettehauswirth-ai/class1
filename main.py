"""
FastAPI NLP Micro-API
─────────────────────
Three lightweight endpoints for health checks, LLM-powered text
summarization, and LLM-powered sentiment analysis — both via Claude.
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
app = FastAPI(
    title="NLP Micro-API",
    description=(
        "A tiny but mighty API that summarizes text and analyses sentiment "
        "— both powered by Claude. Brought to you by Anthropic."
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
    custom_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Optional custom instruction for the summarizer. "
            "Use {max_length} as a placeholder for the sentence limit. "
            "The text to summarize is appended automatically. "
            'Example: "Summarize in {max_length} bullet points, focusing on financial impact."'
        ),
    )

class SummarizeResponse(BaseModel):
    original_length: int
    summary_length: int
    compression_ratio: float
    summary: str
    prompt_used: str

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text to analyse.")
    custom_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Optional custom instruction for the sentiment analyser. "
            "The text to analyse is appended automatically. "
            "The model must still return the required JSON schema. "
            'Example: "Analyse sentiment from the perspective of a brand manager."'
        ),
    )

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    explanation: str
    word_count: int
    highlights: dict
    prompt_used: str


# ──────────────────────────────────────────────
# Helpers — LLM-powered summarizer (Claude)
# ──────────────────────────────────────────────
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


def _get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY environment variable is not set.",
        )
    return key


async def summarize_text(text: str, max_sentences: int = 3, custom_prompt: str | None = None) -> tuple[str, str]:
    """Call the Anthropic API to produce an abstractive summary.
    Returns (summary, prompt_used)."""
    api_key = _get_api_key()

    default_instruction = (
        f"Summarize the following text in at most {max_sentences} sentence(s). "
        "Be concise and capture the key points. Return ONLY the summary, "
        "no preamble or labels."
    )

    if custom_prompt:
        instruction = custom_prompt.replace("{max_length}", str(max_sentences))
    else:
        instruction = default_instruction

    prompt = f"{instruction}\n\n{text}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
        )

    if resp.status_code != 200:
        logger.error("Anthropic API error %s: %s", resp.status_code, resp.text)
        raise HTTPException(
            status_code=502,
            detail=f"Upstream LLM error ({resp.status_code}). Please try again.",
        )

    data = resp.json()
    try:
        return data["content"][0]["text"].strip(), instruction
    except (KeyError, IndexError):
        raise HTTPException(status_code=502, detail="Unexpected LLM response format.")


# ──────────────────────────────────────────────
# Helpers — LLM-powered sentiment analyser
# ──────────────────────────────────────────────
async def analyze_sentiment(text: str, custom_prompt: str | None = None) -> SentimentResponse:
    """Call the Anthropic API to analyse sentiment."""
    api_key = _get_api_key()

    word_count = len(text.split())
    if word_count == 0:
        raise HTTPException(status_code=422, detail="Text contains no analysable words.")

    json_schema = (
        "Respond with ONLY a valid JSON object (no markdown, no backticks) "
        "using exactly this schema:\n"
        "{\n"
        '  "sentiment": "positive" | "negative" | "neutral",\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "explanation": "<1-2 sentence explanation of why>",\n'
        '  "highlights": {\n'
        '    "positive": ["<word or phrase>", ...],\n'
        '    "negative": ["<word or phrase>", ...]\n'
        "  }\n"
        "}"
    )

    default_instruction = "Analyse the sentiment of the following text."

    if custom_prompt:
        instruction = f"{custom_prompt}\n\n{json_schema}"
    else:
        instruction = f"{default_instruction}\n\n{json_schema}"

    prompt = f"{instruction}\n\nText to analyse:\n{text}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
        )

    if resp.status_code != 200:
        logger.error("Anthropic API error %s: %s", resp.status_code, resp.text)
        raise HTTPException(
            status_code=502,
            detail=f"Upstream LLM error ({resp.status_code}). Please try again.",
        )

    data = resp.json()
    try:
        raw_text = data["content"][0]["text"].strip()
    except (KeyError, IndexError):
        raise HTTPException(status_code=502, detail="Unexpected LLM response format.")

    # Strip markdown fences if the model wraps them anyway
    cleaned = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM JSON: %s", raw_text)
        raise HTTPException(status_code=502, detail="LLM returned invalid JSON.")

    return SentimentResponse(
        sentiment=result.get("sentiment", "neutral"),
        confidence=round(float(result.get("confidence", 0.5)), 2),
        explanation=result.get("explanation", ""),
        word_count=word_count,
        highlights=result.get("highlights", {"positive": [], "negative": []}),
        prompt_used=instruction,
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
async def summarize(req: SummarizeRequest):
    """
    LLM-powered text summarizer.

    Sends the input text to Claude (Anthropic API) and returns a concise
    abstractive summary limited to the requested number of sentences.
    Requires ANTHROPIC_API_KEY to be set as an environment variable.
    """
    summary, prompt_used = await summarize_text(
        req.text, max_sentences=req.max_length, custom_prompt=req.custom_prompt
    )
    orig_len = len(req.text)
    summ_len = len(summary)
    return SummarizeResponse(
        original_length=orig_len,
        summary_length=summ_len,
        compression_ratio=round(summ_len / orig_len, 2) if orig_len else 1.0,
        summary=summary,
        prompt_used=prompt_used,
    )


@app.post("/analyze-sentiment", response_model=SentimentResponse, tags=["NLP"])
async def sentiment(req: SentimentRequest):
    """
    LLM-powered sentiment analyser.

    Sends the input text to Claude (Anthropic API) which returns a structured
    JSON response with sentiment label, confidence score, explanation, and
    the specific words/phrases that drove the decision.
    Requires ANTHROPIC_API_KEY to be set as an environment variable.
    """
    return await analyze_sentiment(req.text, custom_prompt=req.custom_prompt)
