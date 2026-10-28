import re

from datasets import load_dataset


def explore_dataset(dataset):
    for split_name in ["train", "test"]:
        split = dataset[split_name]
        labels = split["label"]

        print(f"\n--- {split_name.upper()} ---")
        print(f"Total reviews: {len(split)}")
        print(f"Negative reviews (0): {labels.count(0)}")
        print(f"Positive reviews (1): {labels.count(1)}")


def clean_text(text):
    """Clean an IMDb review while preserving its basic text structure."""
    text = re.sub(r"<br\s*/?>", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def prepare_data(dataset):
    train_texts = [
        clean_text(text) for text in dataset["train"]["text"]
    ]
    train_labels = dataset["train"]["label"]

    test_texts = [
        clean_text(text) for text in dataset["test"]["text"]
    ]
    test_labels = dataset["test"]["label"]

    return train_texts, train_labels, test_texts, test_labels


def main():
    print("Loading IMDb dataset...")

    dataset = load_dataset("stanfordnlp/imdb")

    print("\nDataset loaded successfully!")
    print(dataset)

    explore_dataset(dataset)

    print("\nPreparing dataset...")

    train_texts, train_labels, test_texts, test_labels = prepare_data(dataset)

    print("\nData preparation completed!")
    print(f"Training samples: {len(train_texts)}")
    print(f"Training labels: {len(train_labels)}")
    print(f"Test samples: {len(test_texts)}")
    print(f"Test labels: {len(test_labels)}")


if __name__ == "__main__":
    main()