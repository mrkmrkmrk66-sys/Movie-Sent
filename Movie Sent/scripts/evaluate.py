import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.text_preprocess import load_reviews_csv, prepare_dataset


def ensure_dirs():
	os.makedirs("reports", exist_ok=True)
	os.makedirs("models", exist_ok=True)


def evaluate_lr(df):
	vec_p = os.path.join("models", "lr_vectorizer.joblib")
	model_p = os.path.join("models", "lr_model.joblib")
	if not (os.path.exists(vec_p) and os.path.exists(model_p)):
		print("[LR] Missing artifacts. Train LR first.")
		return
	vectorizer = joblib.load(vec_p)
	model = joblib.load(model_p)
	df_bin = df[df["label"].isin([0, 1])]
	X = df_bin["clean_review"].tolist()
	y = df_bin["label"].astype(int).values
	X_tfidf = vectorizer.transform(X)
	y_pred = model.predict(X_tfidf)
	report = classification_report(y, y_pred, digits=4)
	print("\n[LR] Classification report:\n", report)
	with open(os.path.join("reports", "lr_classification_report.txt"), "w", encoding="utf-8") as f:
		f.write(report)
	cm = confusion_matrix(y, y_pred)
	with open(os.path.join("reports", "confusion_lr.csv"), "w", encoding="utf-8") as f:
		for row in cm:
			f.write(",".join(str(int(v)) for v in row) + "\n")


def evaluate_lstm(df):
	model_p = os.path.join("models", "lstm_model.keras")
	token_p = os.path.join("models", "lstm_tokenizer.joblib")
	if not (os.path.exists(model_p) and os.path.exists(token_p)):
		print("[LSTM] Missing artifacts. Train LSTM first.")
		return
	from tensorflow import keras
	model = keras.models.load_model(model_p)
	tokenizer = joblib.load(token_p)
	df_bin = df[df["label"].isin([0, 1])]
	X = df_bin["clean_review"].astype(str).tolist()
	y = df_bin["label"].astype(int).values
	seq = tokenizer.texts_to_sequences(X)
	X_pad = keras.preprocessing.sequence.pad_sequences(seq, maxlen=200, padding="post", truncating="post")
	probs = model.predict(X_pad, verbose=0).ravel()
	y_pred = (probs >= 0.5).astype(int)
	report = classification_report(y, y_pred, digits=4)
	print("\n[LSTM] Classification report:\n", report)
	with open(os.path.join("reports", "lstm_classification_report.txt"), "w", encoding="utf-8") as f:
		f.write(report)
	cm = confusion_matrix(y, y_pred)
	with open(os.path.join("reports", "confusion_lstm.csv"), "w", encoding="utf-8") as f:
		for row in cm:
			f.write(",".join(str(int(v)) for v in row) + "\n")


def main():
	ensure_dirs()
	df = load_reviews_csv("IMDB Dataset.csv")
	df = prepare_dataset(df)
	evaluate_lr(df)
	evaluate_lstm(df)


if __name__ == "__main__":
	main()
