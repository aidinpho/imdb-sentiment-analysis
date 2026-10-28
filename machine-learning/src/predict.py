from pathlib import Path

import joblib

from load_data import clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.joblib"


def load_model():
    return joblib.load(MODEL_PATH)


def predict_sentiment(text, model):
    cleaned_text = clean_text(text)
    prediction = model.predict([cleaned_text])[0]

    if prediction == 1:
        return "Positive"
    elif prediction == 0:
        return "Negative"

    raise ValueError(f"Unexpected prediction value: {prediction}")


def main():
    model = load_model()
    review = input("Enter a movie review: ")
    sentiment = predict_sentiment(review, model)

    print(f"\nPredicted sentiment: {sentiment}")


if __name__ == "__main__":
    main()