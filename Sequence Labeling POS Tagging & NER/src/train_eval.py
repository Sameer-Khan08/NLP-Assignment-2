from typing import Dict, List, Tuple, Any, Optional

import copy
from xml.parsers.expat import model
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


PAD_LABEL = -100


def masked_flatten_logits_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    logits:         [B, T, C]
    labels:         [B, T]
    attention_mask: [B, T] (bool)

    Returns flattened valid positions only.
    """
    num_labels = logits.size(-1)

    logits_flat = logits.view(-1, num_labels)
    labels_flat = labels.view(-1)
    mask_flat = attention_mask.view(-1)

    valid_logits = logits_flat[mask_flat]
    valid_labels = labels_flat[mask_flat]

    return valid_logits, valid_labels


def compute_token_metrics(
    gold_ids: List[int],
    pred_ids: List[int],
    idx2label: Dict[int, str]
) -> Dict[str, Any]:
    acc = accuracy_score(gold_ids, pred_ids)
    macro_f1 = f1_score(gold_ids, pred_ids, average="macro", zero_division=0)

    gold_labels = [idx2label[i] for i in gold_ids]
    pred_labels = [idx2label[i] for i in pred_ids]

    report = classification_report(
        gold_labels,
        pred_labels,
        zero_division=0,
        output_dict=True
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
    }


def evaluate_sequence_tagger(
    model: nn.Module,
    dataloader,
    device: torch.device,
    idx2label: Dict[int, str],
    criterion: Optional[nn.Module] = None
) -> Dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_batches = 0

    all_gold = []
    all_pred = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            label_ids = batch["label_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lengths = batch["lengths"].to(device)

            logits = model(input_ids, lengths)  # [B, T, C]

            if criterion is not None:
                valid_logits, valid_labels = masked_flatten_logits_labels(
                    logits, label_ids, attention_mask
                )
                loss = criterion(valid_logits, valid_labels)
                total_loss += loss.item()
                total_batches += 1

            preds = torch.argmax(logits, dim=-1)  # [B, T]

            mask = attention_mask.bool()
            gold_valid = label_ids[mask].detach().cpu().tolist()
            pred_valid = preds[mask].detach().cpu().tolist()

            all_gold.extend(gold_valid)
            all_pred.extend(pred_valid)

    metrics = compute_token_metrics(all_gold, all_pred, idx2label)
    avg_loss = total_loss / total_batches if total_batches > 0 else None

    return {
        "loss": avg_loss,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "classification_report": metrics["classification_report"],
        "gold_ids": all_gold,
        "pred_ids": all_pred,
    }


def train_sequence_tagger(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    idx2label: Dict[int, str],
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 30,
    patience: int = 5
) -> Dict[str, Any]:
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    best_state_dict = None

    for epoch in range(1, max_epochs + 1):
        model.train()

        running_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            label_ids = batch["label_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lengths = batch["lengths"].to(device)

            optimizer.zero_grad()

            logits = model(input_ids, lengths)
            valid_logits, valid_labels = masked_flatten_logits_labels(
                logits, label_ids, attention_mask
            )

            loss = criterion(valid_logits, valid_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        train_loss = running_loss / total_batches if total_batches > 0 else 0.0

        val_metrics = evaluate_sequence_tagger(
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

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_macro_f1={val_f1:.4f}"
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


def plot_training_history(history: Dict[str, List[float]], title_prefix: str = "") -> None:
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


def get_confusion_matrix(
    gold_ids: List[int],
    pred_ids: List[int],
    idx2label: Dict[int, str]
):
    labels_sorted = [idx2label[i] for i in sorted(idx2label.keys())]
    gold_labels = [idx2label[i] for i in gold_ids]
    pred_labels = [idx2label[i] for i in pred_ids]

    cm = confusion_matrix(gold_labels, pred_labels, labels=labels_sorted)
    return cm, labels_sorted


def get_most_confused_pairs(
    cm: np.ndarray,
    labels: List[str],
    top_k: int = 3
) -> List[Tuple[str, str, int]]:
    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                pairs.append((labels[i], labels[j], int(cm[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def collect_confusion_examples(
    model: nn.Module,
    dataloader,
    device: torch.device,
    idx2label: Dict[int, str],
    target_gold: str,
    target_pred: str,
    max_examples: int = 2
) -> List[Dict[str, Any]]:
    model.eval()
    examples = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lengths = batch["lengths"].to(device)

            logits = model(input_ids, lengths)
            preds = torch.argmax(logits, dim=-1).cpu()

            label_ids = batch["label_ids"].cpu()
            tokens_batch = batch["tokens"]

            for b_idx in range(len(tokens_batch)):
                tokens = tokens_batch[b_idx]
                gold_seq = label_ids[b_idx].tolist()
                pred_seq = preds[b_idx].tolist()
                mask_seq = batch["attention_mask"][b_idx].tolist()

                gold_labels = []
                pred_labels = []

                for tok, g, p, m in zip(tokens, gold_seq, pred_seq, mask_seq):
                    if not m:
                        continue
                    gold_lab = idx2label[g]
                    pred_lab = idx2label[p]
                    gold_labels.append(gold_lab)
                    pred_labels.append(pred_lab)

                if target_gold in gold_labels and target_pred in pred_labels:
                    mismatch_positions = []
                    for pos, (g_lab, p_lab) in enumerate(zip(gold_labels, pred_labels)):
                        if g_lab == target_gold and p_lab == target_pred:
                            mismatch_positions.append(pos)

                    if mismatch_positions:
                        examples.append({
                            "tokens": tokens,
                            "gold_labels": gold_labels,
                            "pred_labels": pred_labels,
                            "mismatch_positions": mismatch_positions
                        })

                if len(examples) >= max_examples:
                    return examples

    return examples

def train_crf_sequence_tagger(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    idx2label: Dict[int, str],
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 30,
    patience: int = 5
) -> Dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    best_state_dict = None

    for epoch in range(1, max_epochs + 1):
        model.train()

        running_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            label_ids = batch["label_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lengths = batch["lengths"].to(device)

            # IMPORTANT: CRF cannot handle -100 label ids
            safe_label_ids = label_ids.masked_fill(~attention_mask, 0)

            optimizer.zero_grad()

            loss = model.neg_log_likelihood(
                input_ids=input_ids,
                lengths=lengths,
                tags=safe_label_ids,
                mask=attention_mask
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        train_loss = running_loss / total_batches if total_batches > 0 else 0.0

        val_metrics = evaluate_crf_sequence_tagger(
            model=model,
            dataloader=val_loader,
            device=device,
            idx2label=idx2label
        )

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_macro_f1={val_f1:.4f}"
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

def evaluate_crf_sequence_tagger(
    model: nn.Module,
    dataloader,
    device: torch.device,
    idx2label: Dict[int, str]
) -> Dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_batches = 0

    all_gold = []
    all_pred = []
    decoded_sequences = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            label_ids = batch["label_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lengths = batch["lengths"].to(device)

            # IMPORTANT: CRF cannot handle -100 label ids
            safe_label_ids = label_ids.masked_fill(~attention_mask, 0)

            loss = model.neg_log_likelihood(
                input_ids=input_ids,
                lengths=lengths,
                tags=safe_label_ids,
                mask=attention_mask
            )

            decoded = model.decode(
                input_ids=input_ids,
                lengths=lengths,
                mask=attention_mask
            )

            total_loss += loss.item()
            total_batches += 1

            for b_idx, pred_seq in enumerate(decoded):
                seq_len = int(lengths[b_idx].item())
                gold_seq = label_ids[b_idx][:seq_len].detach().cpu().tolist()

                all_gold.extend(gold_seq)
                all_pred.extend(pred_seq)
                decoded_sequences.append(pred_seq)

    metrics = compute_token_metrics(all_gold, all_pred, idx2label)
    avg_loss = total_loss / total_batches if total_batches > 0 else None

    return {
        "loss": avg_loss,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "classification_report": metrics["classification_report"],
        "gold_ids": all_gold,
        "pred_ids": all_pred,
        "decoded_sequences": decoded_sequences,
    }

def extract_entities_from_bio(tags: List[str]) -> List[Tuple[str, int, int]]:
    """
    Returns list of (entity_type, start_idx, end_idx) with end exclusive.
    """
    entities = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            ent_type = tag[2:]
            start = i
            i += 1
            while i < len(tags) and tags[i] == f"I-{ent_type}":
                i += 1
            end = i
            entities.append((ent_type, start, end))
        else:
            i += 1
    return entities


def compute_entity_level_metrics(
    gold_tag_sequences: List[List[str]],
    pred_tag_sequences: List[List[str]],
    entity_types: List[str] = None
) -> Dict[str, Any]:
    if entity_types is None:
        entity_types = ["PER", "LOC", "ORG", "MISC"]

    results = {}
    overall_tp = overall_fp = overall_fn = 0

    for ent_type in entity_types:
        tp = fp = fn = 0

        for gold_tags, pred_tags in zip(gold_tag_sequences, pred_tag_sequences):
            gold_entities = {e for e in extract_entities_from_bio(gold_tags) if e[0] == ent_type}
            pred_entities = {e for e in extract_entities_from_bio(pred_tags) if e[0] == ent_type}

            tp += len(gold_entities & pred_entities)
            fp += len(pred_entities - gold_entities)
            fn += len(gold_entities - pred_entities)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[ent_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0 else 0.0
    )

    results["OVERALL"] = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "tp": overall_tp,
        "fp": overall_fp,
        "fn": overall_fn
    }

    return results


def predict_softmax_sequences(
    model: nn.Module,
    dataloader,
    device: torch.device,
    idx2label: Dict[int, str]
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    model.eval()

    gold_sequences = []
    pred_sequences = []
    token_sequences = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)

            logits = model(input_ids, lengths)
            preds = torch.argmax(logits, dim=-1).cpu()

            label_ids = batch["label_ids"].cpu()
            tokens_batch = batch["tokens"]

            for b_idx, tokens in enumerate(tokens_batch):
                seq_len = len(tokens)
                gold_ids = label_ids[b_idx][:seq_len].tolist()
                pred_ids = preds[b_idx][:seq_len].tolist()

                gold_tags = [idx2label[i] for i in gold_ids]
                pred_tags = [idx2label[i] for i in pred_ids]

                gold_sequences.append(gold_tags)
                pred_sequences.append(pred_tags)
                token_sequences.append(tokens)

    return gold_sequences, pred_sequences, token_sequences


def predict_crf_sequences(
    model: nn.Module,
    dataloader,
    device: torch.device,
    idx2label: Dict[int, str]
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    model.eval()

    gold_sequences = []
    pred_sequences = []
    token_sequences = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lengths = batch["lengths"].to(device)
            label_ids = batch["label_ids"].cpu()
            tokens_batch = batch["tokens"]

            decoded = model.decode(
                input_ids=input_ids,
                lengths=lengths,
                mask=attention_mask
            )

            for b_idx, tokens in enumerate(tokens_batch):
                seq_len = len(tokens)
                gold_ids = label_ids[b_idx][:seq_len].tolist()
                pred_ids = decoded[b_idx]

                gold_tags = [idx2label[i] for i in gold_ids]
                pred_tags = [idx2label[i] for i in pred_ids]

                gold_sequences.append(gold_tags)
                pred_sequences.append(pred_tags)
                token_sequences.append(tokens)

    return gold_sequences, pred_sequences, token_sequences


def collect_ner_error_examples(
    gold_sequences: List[List[str]],
    pred_sequences: List[List[str]],
    token_sequences: List[List[str]],
    error_type: str = "fp",
    max_examples: int = 5
) -> List[Dict[str, Any]]:
    examples = []

    for tokens, gold_tags, pred_tags in zip(token_sequences, gold_sequences, pred_sequences):
        gold_entities = set(extract_entities_from_bio(gold_tags))
        pred_entities = set(extract_entities_from_bio(pred_tags))

        if error_type == "fp":
            errors = list(pred_entities - gold_entities)
        else:
            errors = list(gold_entities - pred_entities)

        for ent in errors:
            examples.append({
                "tokens": tokens,
                "gold_tags": gold_tags,
                "pred_tags": pred_tags,
                "entity": ent
            })
            if len(examples) >= max_examples:
                return examples

    return examples