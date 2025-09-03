import re
import string
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Lightweight, dependency-free text cleaning/tokenization

_stopwords = set(ENGLISH_STOP_WORDS)


def clean_html(text: str) -> str:
	# Remove simple HTML tags
	return re.sub(r"<[^>]+>", " ", text)


def normalize_text(text: str) -> str:
	text = str(text).lower()
	text = clean_html(text)
	text = re.sub(r"https?://\S+|www\.\S+", " ", text)
	text = re.sub(r"[@#]\w+", " ", text)
	text = re.sub(r"\d+", " ", text)
	text = text.translate(str.maketrans('', '', string.punctuation))
	text = re.sub(r"\s+", " ", text).strip()
	return text


def simple_tokenize(text: str) -> List[str]:
	# Keep alphabetic tokens and apostrophes within words
	return re.findall(r"[a-zA-Z']+", text)


def preprocess_text(text: str) -> str:
	text = normalize_text(text)
	tokens = simple_tokenize(text)
	processed = [t for t in tokens if len(t) > 2 and t not in _stopwords]
	return " ".join(processed)


def load_reviews_csv(path: str) -> pd.DataFrame:
	# Robust CSV loading: utf-8, python engine, skip bad lines
	df = pd.read_csv(path, encoding='utf-8', engine='python', on_bad_lines='skip')
	df.columns = [c.strip().lower() for c in df.columns]
	review_col = None
	sent_col = None
	for c in df.columns:
		if c in {"review", "text", "content", "review_text"}:
			review_col = c
		if c in {"sentiment", "label", "target"}:
			sent_col = c
	if review_col is None:
		raise ValueError("Could not find review text column in CSV. Expected one of: review, text, content, review_text")
	if sent_col is None:
		raise ValueError("Could not find sentiment/label column in CSV. Expected one of: sentiment, label, target")
	df = df[[review_col, sent_col] + [c for c in df.columns if c not in {review_col, sent_col}]]
	df = df.rename(columns={review_col: "review", sent_col: "sentiment"})
	df = df.dropna(subset=["review", "sentiment"]).drop_duplicates(subset=["review"])  # type: ignore
	return df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df["clean_review"] = df["review"].astype(str).apply(preprocess_text)
	label_map = {"positive": 1, "pos": 1, "neg": 0, "negative": 0, 1: 1, 0: 0, "neutral": 2}
	df["label"] = df["sentiment"].map(lambda x: label_map.get(str(x).lower(), x))
	return df
