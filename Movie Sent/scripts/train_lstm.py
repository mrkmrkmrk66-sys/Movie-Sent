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
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from src.utils.text_preprocess import load_reviews_csv, prepare_dataset

RANDOM_STATE = 42
TEST_SIZE = 0.30
MAX_VOCAB = 30000
MAX_LEN = 200
EMBED_DIM = 100
LSTM_UNITS = 128
EPOCHS = 3
BATCH_SIZE = 64


def build_model(vocab_size: int):
	inputs = keras.Input(shape=(MAX_LEN,), dtype="int32")
	x = layers.Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN)(inputs)
	x = layers.SpatialDropout1D(0.2)(x)
	x = layers.LSTM(LSTM_UNITS, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(x)
	x = layers.Dense(64, activation="relu")(x)
	x = layers.Dropout(0.3)(x)
	outputs = layers.Dense(1, activation="sigmoid")(x)
	model = keras.Model(inputs, outputs)
	model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(2e-4), metrics=["accuracy"])
	return model


def main():
	data_path = os.path.join(os.getcwd(), "IMDB Dataset.csv")
	df = load_reviews_csv(data_path)
	df = prepare_dataset(df)
	df = df[df["label"].isin([0, 1])]

	X = df["clean_review"].astype(str).tolist()
	y = df["label"].astype(int).values

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
	)

	tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
	tokenizer.fit_on_texts(X_train)

	def to_seq(texts):
		seq = tokenizer.texts_to_sequences(texts)
		return keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

	X_train_seq = to_seq(X_train)
	X_test_seq = to_seq(X_test)

	model = build_model(vocab_size=MAX_VOCAB)

	es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)
	os.makedirs("models", exist_ok=True)
	ckpt_path = os.path.join("models", "lstm_checkpoint.keras")
	mc = keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True)

	model.fit(
		X_train_seq,
		y_train,
		validation_data=(X_test_seq, y_test),
		epochs=EPOCHS,
		batch_size=BATCH_SIZE,
		callbacks=[es, mc],
		verbose=1,
	)

	model.save(os.path.join("models", "lstm_model.keras"))
	joblib.dump(tokenizer, os.path.join("models", "lstm_tokenizer.joblib"))

	meta = {
		"max_vocab": MAX_VOCAB,
		"max_len": MAX_LEN,
		"embed_dim": EMBED_DIM,
		"lstm_units": LSTM_UNITS,
		"epochs": EPOCHS,
		"batch_size": BATCH_SIZE,
		"test_size": TEST_SIZE,
		"random_state": RANDOM_STATE,
	}
	with open(os.path.join("models", "lstm_meta.json"), "w", encoding="utf-8") as f:
		json.dump(meta, f, indent=2)


if __name__ == "__main__":
	main()
