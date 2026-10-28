import re

from nltk.tokenize import word_tokenize


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def clean_text(text):
    """
    Remove HTML tags and normalize whitespace.
    """
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text):
    """
    Clean text and convert it into lowercase tokens.
    """
    text = clean_text(text)
    tokens = word_tokenize(text.lower())

    return tokens


def encode(text, vocab):
    """
    Convert text into token IDs using the vocabulary.
    """
    tokens = tokenize(text)

    token_ids = [
        vocab.get(token, vocab[UNK_TOKEN])
        for token in tokens
    ]

    return token_ids


def pad_or_truncate(token_ids, max_length=300, pad_token_id=0):
    """
    Truncate long sequences and pad short sequences.
    """
    token_ids = token_ids[:max_length]

    padding_length = max_length - len(token_ids)

    token_ids = token_ids + [pad_token_id] * padding_length

    return token_ids