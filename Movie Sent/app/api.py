from flask import Flask, request, jsonify
import os
import sys
import joblib

# Ensure project root is on sys.path when running this file directly
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.utils.text_preprocess import preprocess_text

app = Flask(__name__)

# Enable CORS for local frontend
try:
	from flask_cors import CORS
	CORS(app)
except Exception:
	pass

@app.route("/", methods=["GET"])
def root():
	return jsonify({
		"message": "MovieSent API running",
		"endpoints": {
			"health": "/health",
			"predict": {"path": "/predict", "method": "POST", "body": {"text": "your review", "model": "lr|lstm"}}
		}
	})

LR_VECTORIZER_P = os.path.join("models", "lr_vectorizer.joblib")
LR_MODEL_P = os.path.join("models", "lr_model.joblib")
LSTM_MODEL_P = os.path.join("models", "lstm_model.keras")
LSTM_TOKENIZER_P = os.path.join("models", "lstm_tokenizer.joblib")

lr_vectorizer = joblib.load(LR_VECTORIZER_P) if os.path.exists(LR_VECTORIZER_P) else None
lr_model = joblib.load(LR_MODEL_P) if os.path.exists(LR_MODEL_P) else None

lstm_model = None
lstm_tokenizer = None
try:
	from tensorflow import keras
	if os.path.exists(LSTM_MODEL_P) and os.path.exists(LSTM_TOKENIZER_P):
		lstm_model = keras.models.load_model(LSTM_MODEL_P)
		lstm_tokenizer = joblib.load(LSTM_TOKENIZER_P)
except Exception:
	pass

@app.route("/health", methods=["GET"])
def health():
	return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
	payload = request.get_json(force=True)
	text = payload.get("text", "").strip()
	model_choice = payload.get("model", "lr").lower()
	if not text:
		return jsonify({"error": "text is required"}), 400

	# Preprocess consistently with training
	cleaned = preprocess_text(text)

	if model_choice == "lr":
		if lr_vectorizer is None or lr_model is None:
			return jsonify({"error": "LR artifacts not found. Train LR first."}), 500
		X = lr_vectorizer.transform([cleaned])
		pred = int(lr_model.predict(X)[0])
		prob = float(lr_model.predict_proba(X)[0][pred]) if hasattr(lr_model, 'predict_proba') else None
		label = "positive" if pred == 1 else "negative"
		return jsonify({"model": "lr", "label": label, "probability": prob})
	elif model_choice == "lstm":
		if lstm_model is None or lstm_tokenizer is None:
			return jsonify({"error": "LSTM artifacts not found. Train LSTM first."}), 500
		seq = lstm_tokenizer.texts_to_sequences([cleaned])
		from tensorflow import keras as k
		pad = k.preprocessing.sequence.pad_sequences(seq, maxlen=200, padding="post", truncating="post")
		p = float(lstm_model.predict(pad, verbose=0).ravel()[0])
		label = "positive" if p >= 0.5 else "negative"
		return jsonify({"model": "lstm", "label": label, "probability": p})
	else:
		return jsonify({"error": "Unknown model. Use 'lr' or 'lstm'."}), 400

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)
