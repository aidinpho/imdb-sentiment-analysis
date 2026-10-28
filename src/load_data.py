from datasets import load_dataset


def load_imdb_dataset():
    """
    Load the IMDb movie review dataset.

    Returns:
        DatasetDict: IMDb dataset containing train and test splits.
    """
    dataset = load_dataset("stanfordnlp/imdb")
    return dataset