import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_financial_text(text: str) -> str:
    """
    Clean raw financial text while preserving numbers, currency symbols,
    percentages, and punctuation needed by Transformer tokenizers.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove @mentions but keep $TICKERS (strip only the '$' prefix so ticker letters remain)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\$([A-Za-z]{1,6})\b", r"\1", text)
    # Collapse repeated whitespace/newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extractive_summary(text: str, num_sentences: int = 3) -> str:
    """
    Generate a fast extractive summary for multi-sentence documents using TF-IDF
    sentence scoring (no extra heavy model download required).
    """
    cleaned = clean_financial_text(text)
    if not cleaned:
        return ""

    # Split into sentences on '.', '!', or '?'
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if len(s.strip()) > 20]

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).ravel()

    top_indices = sorted(np.argsort(sentence_scores)[-num_sentences:])
    return " ".join(sentences[i] for i in top_indices)