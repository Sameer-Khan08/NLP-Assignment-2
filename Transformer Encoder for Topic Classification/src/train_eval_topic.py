from typing import Dict, List, Any, Tuple

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import time

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C]
        target: [B]
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.epsilon / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.epsilon)

        loss = torch.sum(-true_dist * log_probs, dim=-1).mean()
        return loss


def evaluate_topic_classifier(
    model: nn.Module,
    dataloader,
    device: torch.device,
    idx2label: Dict[int, str],
    criterion: nn.Module = None
) -> Dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_batches = 0

    all_gold = []
    all_pred = []
    stored_attention_maps = []
    stored_batch = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, attn_maps = model(input_ids, attention_mask)

            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item()
                total_batches += 1

            preds = torch.argmax(logits, dim=-1)

            all_gold.extend(labels.detach().cpu().tolist())
            all_pred.extend(preds.detach().cpu().tolist())

            if batch_idx == 0:
                stored_attention_maps = [a.detach().cpu() for a in attn_maps]
                stored_batch = {
                    "tokens": batch["tokens"],
                    "labels": batch["label"].detach().cpu().tolist(),
                    "titles": batch["title"],
                    "doc_id": batch["doc_id"]
                }

    acc = accuracy_score(all_gold, all_pred)
    macro_f1 = f1_score(all_gold, all_pred, average="macro", zero_division=0)

    gold_labels = [idx2label[i] for i in all_gold]
    pred_labels = [idx2label[i] for i in all_pred]

    report = classification_report(
        gold_labels,
        pred_labels,
        zero_division=0,
        output_dict=True
    )

    avg_loss = total_loss / total_batches if total_batches > 0 else None

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "gold_ids": all_gold,
        "pred_ids": all_pred,
        "classification_report": report,
        "attention_maps": stored_attention_maps,
        "attention_batch": stored_batch,
    }


def train_topic_classifier(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    idx2label: Dict[int, str],
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 25,
    patience: int = 5,
    label_smoothing: float = 0.1
) -> Dict[str, Any]:
    criterion = LabelSmoothingCrossEntropy(epsilon=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "epoch_time_sec": [],
    }

    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    best_state_dict = None

    for epoch in range(1, max_epochs + 1):
        start_time = time.time()
        model.train()

        running_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        epoch_time = time.time() - start_time
        train_loss = running_loss / total_batches if total_batches > 0 else 0.0

        val_metrics = evaluate_topic_classifier(
            model=model,
            dataloader=val_loader,
            device=device,
            idx2label=idx2label,
            criterion=criterion
        )

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["epoch_time_sec"].append(epoch_time)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_macro_f1={val_f1:.4f} | "
            f"time={epoch_time:.2f}s"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "model": model,
        "history": history,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
    }

def plot_topic_training_history(history: Dict[str, List[float]], title_prefix: str = "") -> None:
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    plt.title(f"{title_prefix} Loss per Epoch".strip())
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["val_accuracy"], marker="o", label="Validation Accuracy")
    plt.plot(epochs, history["val_macro_f1"], marker="o", label="Validation Macro-F1")
    plt.title(f"{title_prefix} Validation Metrics per Epoch".strip())
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def get_topic_confusion_matrix(
    gold_ids: List[int],
    pred_ids: List[int],
    idx2label: Dict[int, str]
) -> Tuple[np.ndarray, List[str]]:
    labels_sorted = [idx2label[i] for i in sorted(idx2label.keys())]
    gold_labels = [idx2label[i] for i in gold_ids]
    pred_labels = [idx2label[i] for i in pred_ids]
    cm = confusion_matrix(gold_labels, pred_labels, labels=labels_sorted)
    return cm, labels_sorted


def attention_matrix_for_sample(attention_maps: List[torch.Tensor], layer_idx: int = 0, head_idx: int = 0, sample_idx: int = 0):
    """
    attention_maps[layer]: [B, H, T, T]
    """
    return attention_maps[layer_idx][sample_idx, head_idx].numpy()




def train_topic_classifier_simple_ce(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    idx2label: Dict[int, str],
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 25,
    patience: int = 5
) -> Dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "epoch_time_sec": [],
    }

    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    best_state_dict = None

    for epoch in range(1, max_epochs + 1):
        start_time = time.time()
        model.train()

        running_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        epoch_time = time.time() - start_time
        train_loss = running_loss / total_batches if total_batches > 0 else 0.0

        val_metrics = evaluate_topic_classifier_non_transformer(
            model=model,
            dataloader=val_loader,
            device=device,
            idx2label=idx2label,
            criterion=criterion
        )

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["epoch_time_sec"].append(epoch_time)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_macro_f1={val_f1:.4f} | "
            f"time={epoch_time:.2f}s"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "model": model,
        "history": history,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
    }


def evaluate_topic_classifier_non_transformer(
    model: nn.Module,
    dataloader,
    device: torch.device,
    idx2label: Dict[int, str],
    criterion: nn.Module = None
) -> Dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_batches = 0

    all_gold = []
    all_pred = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)

            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item()
                total_batches += 1

            preds = torch.argmax(logits, dim=-1)

            all_gold.extend(labels.detach().cpu().tolist())
            all_pred.extend(preds.detach().cpu().tolist())

    acc = accuracy_score(all_gold, all_pred)
    macro_f1 = f1_score(all_gold, all_pred, average="macro", zero_division=0)

    avg_loss = total_loss / total_batches if total_batches > 0 else None

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "gold_ids": all_gold,
        "pred_ids": all_pred,
    }