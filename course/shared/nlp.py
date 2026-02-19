"""NLP utilities for financial text analysis.

Provides text cleaning for financial documents (SEC filings, earnings calls)
and interface stubs for FinBERT sentiment and text embeddings.

First introduced in Week 7; imported by later weeks that combine textual
and numerical features.

External deps: transformers, sentence-transformers (imported lazily).
"""
from __future__ import annotations

import re

import numpy as np


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_financial_text(text: str) -> str:
    """Clean financial document text for NLP processing.

    Strips common boilerplate patterns, normalizes whitespace, and handles
    artifacts typical of SEC filings and earnings transcripts.
    """
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove common SEC boilerplate markers
    text = re.sub(r"(?i)(exhibit\s+\d+|form\s+\d+-[a-z]+)", "", text)
    # Remove page numbers and headers
    text = re.sub(r"\n\s*-\s*\d+\s*-\s*\n", "\n", text)
    # Normalize quotes and dashes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split financial text into sentences.

    Handles common abbreviations (Inc., Corp., No., vs.) that would
    cause false splits with naive period-splitting.
    """
    abbreviations = r"(?:Inc|Corp|Ltd|No|vs|Mr|Mrs|Dr|Jr|Sr|etc|approx)"
    # Split on period/question/exclamation followed by space + uppercase
    # but not after known abbreviations
    pattern = rf"(?<!{abbreviations})\.\s+(?=[A-Z])"
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Sentiment extraction (stubs — require transformers)
# ---------------------------------------------------------------------------

def extract_sentiment_finbert(
    texts: list[str],
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 32,
) -> list[dict]:
    """Extract sentiment scores using FinBERT.

    Args:
        texts: list of text strings.
        model_name: HuggingFace model identifier.
        batch_size: inference batch size.

    Returns:
        List of dicts with keys: label (positive/negative/neutral), score.

    Stub — implement when Week 7 blueprint is finalized.
    Requires: pip install transformers torch
    """
    raise NotImplementedError(
        "Stub — implement when Week 7 blueprint is finalized. "
        "Expected implementation: transformers pipeline('sentiment-analysis', "
        f"model='{model_name}') over batches."
    )


def embed_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Embed texts using sentence-transformers.

    Args:
        texts: list of text strings.
        model_name: sentence-transformers model identifier.

    Returns:
        (n_texts, embedding_dim) numpy array.

    Stub — implement when Week 7 blueprint is finalized.
    Requires: pip install sentence-transformers
    """
    raise NotImplementedError(
        "Stub — implement when Week 7 blueprint is finalized. "
        "Expected implementation: SentenceTransformer(model_name).encode(texts)"
    )
