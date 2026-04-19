import json
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_LABEL = -100  # ignored by CrossEntropyLoss


def load_word2idx(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)


def build_model_vocab(word2idx: Dict[str, int], embeddings: np.ndarray) -> Tuple[Dict[str, int], Dict[int, str], np.ndarray]:
    """
    Adds <PAD> token at index 0 and shifts all existing ids by +1.
    Embedding matrix is expanded with a zero row for PAD.
    """
    shifted_word2idx = {word: idx + 1 for word, idx in word2idx.items()}
    shifted_word2idx[PAD_TOKEN] = 0

    idx2word = {idx: word for word, idx in shifted_word2idx.items()}

    pad_row = np.zeros((1, embeddings.shape[1]), dtype=np.float32)
    expanded_embeddings = np.vstack([pad_row, embeddings.astype(np.float32)])

    return shifted_word2idx, idx2word, expanded_embeddings


def tokens_to_ids(tokens: List[str], word2idx: Dict[str, int]) -> List[int]:
    unk_id = word2idx[UNK_TOKEN]
    return [word2idx.get(tok, unk_id) for tok in tokens]


def labels_to_ids(labels: List[str], label2idx: Dict[str, int]) -> List[int]:
    return [label2idx[label] for label in labels]


class SequenceTaggingDataset(Dataset):
    def __init__(
        self,
        examples: List[Dict[str, Any]],
        word2idx: Dict[str, int],
        label2idx: Dict[str, int],
        task: str = "pos"
    ):
        assert task in {"pos", "ner"}, "task must be 'pos' or 'ner'"
        self.examples = examples
        self.word2idx = word2idx
        self.label2idx = label2idx
        self.task = task

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        tokens = ex["tokens"]
        labels = ex["pos_tags"] if self.task == "pos" else ex["ner_tags"]

        input_ids = tokens_to_ids(tokens, self.word2idx)
        label_ids = labels_to_ids(labels, self.label2idx)

        return {
            "tokens": tokens,
            "labels": labels,
            "input_ids": input_ids,
            "label_ids": label_ids,
            "length": len(tokens),
            "topic": ex["topic"],
            "sentence_id": ex["sentence_id"],
        }


def pad_batch(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    lengths = [item["length"] for item in batch]
    max_len = max(lengths)

    batch_input_ids = []
    batch_label_ids = []
    batch_attention_mask = []
    batch_tokens = []
    batch_labels = []
    batch_topics = []
    batch_sentence_ids = []

    for item in batch:
        seq_len = item["length"]
        pad_len = max_len - seq_len

        padded_input_ids = item["input_ids"] + [pad_token_id] * pad_len
        padded_label_ids = item["label_ids"] + [PAD_LABEL] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len

        batch_input_ids.append(padded_input_ids)
        batch_label_ids.append(padded_label_ids)
        batch_attention_mask.append(attention_mask)
        batch_tokens.append(item["tokens"])
        batch_labels.append(item["labels"])
        batch_topics.append(item["topic"])
        batch_sentence_ids.append(item["sentence_id"])

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "label_ids": torch.tensor(batch_label_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.bool),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "tokens": batch_tokens,
        "labels": batch_labels,
        "topics": batch_topics,
        "sentence_ids": batch_sentence_ids,
    }


def make_collate_fn(pad_token_id: int):
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return pad_batch(batch, pad_token_id=pad_token_id)
    return collate_fn