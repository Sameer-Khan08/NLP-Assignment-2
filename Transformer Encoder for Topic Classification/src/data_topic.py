import os
import re
import json
import random
from typing import Dict, List, Tuple, Any

import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
random.seed(SEED)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"


TOPIC_KEYWORDS = {
    "politics": [
        "حکومت", "وزیر", "وزیراعظم", "پارلیمان", "اسمبلی",
        "انتخاب", "انتخابات", "سیاسی", "عمران", "نواز", "شہباز", "تحریک"
    ],
    "sports": [
        "کرکٹ", "میچ", "ٹیم", "کھلاڑی", "وکٹ", "رنز", "کپ", "ورلڈ"
    ],
    "economy": [
        "معیشت", "بینک", "بجٹ", "روپے", "ڈالر", "حصص", "سٹاک",
        "سرمایہ", "نجکاری", "تجارت"
    ],
    "international": [
        "امریکہ", "روس", "چین", "ایران", "بھارت", "انڈیا", "غزہ",
        "اقوام", "متحدہ", "بین الاقوامی", "سفارتی", "معاہدہ"
    ],
    "health_society": [
        "ہسپتال", "ڈاکٹر", "صحت", "تعلیم", "طالبہ", "یونیورسٹی",
        "بارش", "برفباری", "بیماری", "مریض", "خودکشی"
    ]
}


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_numbered_blocks(text: str) -> Dict[str, str]:
    pattern = re.compile(r"\[(\d+)\]\s*(.*?)(?=\n\[\d+\]\s*|\Z)", re.DOTALL)
    blocks: Dict[str, str] = {}
    for match in pattern.finditer(text):
        doc_id = match.group(1)
        content = match.group(2).strip()
        blocks[doc_id] = content
    return blocks


def extract_cleaned_sentences(block_text: str) -> List[List[str]]:
    sentences = re.findall(r"<SOS>\s*(.*?)\s*<EOS>", block_text, flags=re.DOTALL)
    out: List[List[str]] = []
    for sent in sentences:
        tokens = [tok.strip() for tok in sent.split() if tok.strip()]
        if tokens:
            out.append(tokens)
    return out


def assign_topic(title: str, tokens: List[str]) -> str:
    text = (title + " " + " ".join(tokens[:250])).lower()
    scores: Dict[str, int] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        scores[topic] = sum(text.count(kw.lower()) for kw in keywords)

    best_topic = max(scores, key=scores.get)
    if scores[best_topic] == 0:
        return "health_society"
    return best_topic


def load_documents(cleaned_path: str, metadata_path: str) -> List[Dict[str, Any]]:
    cleaned_text = read_text(cleaned_path)
    metadata = read_json(metadata_path)

    cleaned_blocks = parse_numbered_blocks(cleaned_text)
    common_ids = sorted(
        set(cleaned_blocks.keys()) & set(metadata.keys()),
        key=lambda x: int(x)
    )

    docs = []
    for doc_id in common_ids:
        cleaned_sentences = extract_cleaned_sentences(cleaned_blocks[doc_id])
        cleaned_tokens = [tok for sent in cleaned_sentences for tok in sent]

        title = metadata[doc_id].get("title", "")
        topic = assign_topic(title, cleaned_tokens)

        docs.append({
            "id": doc_id,
            "title": title,
            "tokens": cleaned_tokens,
            "topic": topic
        })

    return docs


def build_topic_dataframe(docs: List[Dict[str, Any]], min_len: int = 20) -> pd.DataFrame:
    rows = []
    for d in docs:
        if len(d["tokens"]) >= min_len:
            rows.append({
                "doc_id": d["id"],
                "title": d["title"],
                "tokens": d["tokens"],
                "topic": d["topic"]
            })
    return pd.DataFrame(rows)


def stratified_split_topic(df: pd.DataFrame, random_state: int = SEED):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=df["topic"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df["topic"]
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_vocab(token_lists: List[List[str]], max_vocab: int = 10000):
    from collections import Counter

    counter = Counter(tok for seq in token_lists for tok in seq)
    most_common = [w for w, _ in counter.most_common(max_vocab)]

    idx2word = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN] + most_common
    word2idx = {w: i for i, w in enumerate(idx2word)}

    return word2idx, idx2word, counter


def encode_tokens(tokens: List[str], word2idx: Dict[str, int], max_len: int):
    cls_id = word2idx[CLS_TOKEN]
    unk_id = word2idx[UNK_TOKEN]
    pad_id = word2idx[PAD_TOKEN]

    ids = [cls_id] + [word2idx.get(tok, unk_id) for tok in tokens[:max_len - 1]]
    attn_mask = [1] * len(ids)

    if len(ids) < max_len:
        pad_len = max_len - len(ids)
        ids += [pad_id] * pad_len
        attn_mask += [0] * pad_len

    return ids, attn_mask


def build_label_mapping():
    labels = ["politics", "sports", "economy", "international", "health_society"]
    label2idx = {lab: i for i, lab in enumerate(labels)}
    idx2label = {i: lab for lab, i in label2idx.items()}
    return label2idx, idx2label