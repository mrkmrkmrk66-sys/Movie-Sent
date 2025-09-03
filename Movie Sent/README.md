## MovieSent – Dual Approach Sentiment Analysis

A complete end-to-end project that trains and serves two sentiment models over movie reviews:
- Classical TF-IDF + Logistic Regression
- Deep Learning LSTM (Keras/TensorFlow)

### Quickstart

1) Create and activate a Python 3.10+ environment
2) Install requirements

```bash
pip install -r requirements.txt
```

3) Ensure dataset is present

Place your dataset CSV in the project root as `IMDB Dataset.csv` (already provided). It should contain at least these columns:
- `review`: raw text of the review
- `sentiment`: label values like `positive` or `negative`

4) Train models

```bash
# Train Logistic Regression
python scripts/train_logreg.py

# Train LSTM
python scripts/train_lstm.py
```

5) Evaluate models (metrics + confusion matrices)

```bash
python scripts/evaluate.py
```

6) Run the API and open the UI

```bash
# Start Flask API
python app/api.py

# Then open web/index.html in your browser (or serve statically)
```

### Project Structure

- `scripts/`: training and evaluation scripts
- `src/utils/`: reusable data loading and preprocessing utilities
- `app/`: Flask API server to expose prediction endpoints
- `web/`: simple responsive frontend
- `models/`: saved artifacts (created after training)
- `reports/`: metrics and confusion matrices (created after evaluation)

### Notes
- The provided IMDB dataset is binary (positive/negative). Neutral is not included; UI shows two-class predictions.
- You can plug in a 3-class dataset if available; the pipeline will adapt where noted in the code.
