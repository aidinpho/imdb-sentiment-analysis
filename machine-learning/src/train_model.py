from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from load_data import prepare_data
from datasets import load_dataset
from pathlib import Path
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.joblib"


def main():
    print("Loading IMDb dataset...")

    dataset = load_dataset("stanfordnlp/imdb")

    print("Preparing data...")

    train_texts, train_labels, test_texts, test_labels = prepare_data(dataset)

    print(f"Training samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")

    print("\nCreating model pipeline...")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    print("Pipeline created successfully!")

    print("\nTraining model...")

    model.fit(train_texts, train_labels)

    print("Model training completed!")

    print("\nSaving model...")

    joblib.dump(model, MODEL_PATH)

    print("Model saved successfully!")

    print("\nEvaluating model...")

    accuracy = model.score(test_texts, test_labels)

    print(f"Test Accuracy: {accuracy:.4f}")

    print("\nGenerating predictions...")

    predictions = model.predict(test_texts)

    print("\nClassification Report:")
    print(classification_report(
        test_labels,
        predictions,
        target_names=["Negative", "Positive"]
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, predictions))


if __name__ == "__main__":
    main()