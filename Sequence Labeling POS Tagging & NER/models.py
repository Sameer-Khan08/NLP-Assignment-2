from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMEncoder(nn.Module):
    """
    Shared LSTM encoder for sequence tagging.

    Supports:
    - bidirectional or unidirectional LSTM
    - frozen or trainable embeddings
    - pretrained or random embeddings
    - dropout between LSTM layers
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        pad_idx: int = 0,
        embedding_matrix: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        random_init_embeddings: bool = False,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )

        if embedding_matrix is not None and not random_init_embeddings:
            if not isinstance(embedding_matrix, torch.Tensor):
                embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
            self.embedding.weight.data.copy_(embedding_matrix)
        else:
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            with torch.no_grad():
                self.embedding.weight[pad_idx].fill_(0.0)

        self.embedding.weight.requires_grad = not freeze_embeddings

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional
        )

        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

    def forward(self, input_ids, lengths):
        """
        input_ids: [B, T]
        lengths:   [B]
        """
        embeds = self.embedding(input_ids)  # [B, T, E]

        packed = pack_padded_sequence(
            embeds,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, _ = self.lstm(packed)

        lstm_out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True
        )  # [B, T, H*dir]

        return lstm_out


class POSTagger(nn.Module):
    """
    POS model:
    Embedding -> 2-layer BiLSTM -> Linear classifier
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_labels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        pad_idx: int = 0,
        embedding_matrix=None,
        freeze_embeddings: bool = False,
        random_init_embeddings: bool = False,
    ):
        super().__init__()

        self.encoder = BiLSTMEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            pad_idx=pad_idx,
            embedding_matrix=embedding_matrix,
            freeze_embeddings=freeze_embeddings,
            random_init_embeddings=random_init_embeddings,
        )

        self.classifier = nn.Linear(self.encoder.output_dim, num_labels)

    def forward(self, input_ids, lengths):
        encoded = self.encoder(input_ids, lengths)      # [B, T, D]
        logits = self.classifier(encoded)               # [B, T, C]
        return logits


class NERSoftmaxTagger(nn.Module):
    """
    NER baseline without CRF:
    Embedding -> 2-layer BiLSTM -> Linear classifier
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_labels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        pad_idx: int = 0,
        embedding_matrix=None,
        freeze_embeddings: bool = False,
        random_init_embeddings: bool = False,
    ):
        super().__init__()

        self.encoder = BiLSTMEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            pad_idx=pad_idx,
            embedding_matrix=embedding_matrix,
            freeze_embeddings=freeze_embeddings,
            random_init_embeddings=random_init_embeddings,
        )

        self.classifier = nn.Linear(self.encoder.output_dim, num_labels)

    def forward(self, input_ids, lengths):
        encoded = self.encoder(input_ids, lengths)      # [B, T, D]
        emissions = self.classifier(encoded)            # [B, T, C]
        return emissions