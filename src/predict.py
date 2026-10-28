import json
from pathlib import Path

import nltk
import torch

from src.model import SentimentLSTM
from src.preprocess import encode, pad_or_truncate


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "sentiment_lstm.pth"
VOCAB_PATH = MODELS_DIR / "vocab.json"

MAX_LENGTH = 300
EMBEDDING_DIM = 100
HIDDEN_SIZE = 64
NUM_CLASSES = 2


def load_vocab():
    """Load the saved vocabulary."""
    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    return vocab


def load_model(vocab, device):
    """Load the trained sentiment analysis model."""
    model = SentimentLSTM(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        padding_idx=vocab["<PAD>"],
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def predict_sentiment(text, model, vocab, device):
    """Predict whether a movie review is positive or negative."""
    token_ids = encode(text, vocab)

    token_ids = pad_or_truncate(
        token_ids,
        max_length=MAX_LENGTH,
        pad_token_id=vocab["<PAD>"],
    )

    input_tensor = torch.tensor(
        token_ids,
        dtype=torch.long,
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

        predicted_class = torch.argmax(
            probabilities,
            dim=1,
        ).item()

    labels = {
        0: "Negative",
        1: "Positive",
    }

    sentiment = labels[predicted_class]
    score = probabilities[0, predicted_class].item()

    return sentiment, score


def main():
    """Load the model and predict a sample review."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    vocab = load_vocab()
    model = load_model(vocab, device)

    review = input("Enter a movie review: ").strip()

    if not review:
        print("Please enter a review.")
        return

    sentiment, score = predict_sentiment(
        review,
        model,
        vocab,
        device,
    )

    print(f"\nSentiment: {sentiment}")
    print(f"Softmax score: {score:.2%}")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()