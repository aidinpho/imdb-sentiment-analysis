import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SentimentLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_size=64,
        num_classes=2,
        padding_idx=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.classifier = nn.Linear(
            in_features=hidden_size,
            out_features=num_classes,
        )

    def forward(self, x):
        lengths = (x != 0).sum(dim=1)

        embedded = self.embedding(x)

        packed_embedded = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden_state, _) = self.lstm(packed_embedded)

        final_hidden = hidden_state[-1]

        logits = self.classifier(final_hidden)

        return logits