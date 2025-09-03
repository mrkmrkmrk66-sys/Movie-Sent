import os
import sys
import json

# Ensure project root is on sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.utils.text_preprocess import load_reviews_csv, prepare_dataset

RANDOM_STATE = 42
TEST_SIZE = 0.30
NGRAM_RANGE = (1, 3)
MAX_FEATURES = 100000


def main():
	data_path = os.path.join(os.getcwd(), "IMDB Dataset.csv")
	df = load_reviews_csv(data_path)
	df = prepare_dataset(df)
	df = df[df["label"].isin([0, 1])]

	X = df["clean_review"].values
	y = df["label"].astype(int).values

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
	)

	vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES, min_df=2)
	X_train_tfidf = vectorizer.fit_transform(X_train)
	X_test_tfidf = vectorizer.transform(X_test)

	classes = np.unique(y_train)
	class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
	class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

	clf = LogisticRegression(max_iter=1000, class_weight=class_weight_dict)
	clf.fit(X_train_tfidf, y_train)

	y_pred = clf.predict(X_test_tfidf)
	print(classification_report(y_test, y_pred, digits=4))

	os.makedirs("models", exist_ok=True)
	joblib.dump(vectorizer, os.path.join("models", "lr_vectorizer.joblib"))
	joblib.dump(clf, os.path.join("models", "lr_model.joblib"))

	meta = {
		"ngram_range": NGRAM_RANGE,
		"max_features": MAX_FEATURES,
		"test_size": TEST_SIZE,
		"random_state": RANDOM_STATE,
	}
	with open(os.path.join("models", "lr_meta.json"), "w", encoding="utf-8") as f:
		json.dump(meta, f, indent=2)


if __name__ == "__main__":
	main()
