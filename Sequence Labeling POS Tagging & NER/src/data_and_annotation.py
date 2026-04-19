import os
import re
import json
import random
from typing import Dict, List, Tuple, Any

import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
random.seed(SEED)

UNK_TOKEN = "<UNK>"

POS_TAGS = [
    "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET",
    "CONJ", "POST", "NUM", "PUNC", "UNK"
]

NER_TAGS = [
    "B-PER", "I-PER",
    "B-LOC", "I-LOC",
    "B-ORG", "I-ORG",
    "B-MISC", "I-MISC",
    "O"
]

PUNCT_TOKENS = {
    "۔", "،", ".", ",", "!", "؟", ":", ";", "؛",
    "(", ")", "[", "]", "{", "}", "-", "—", "–",
    '"', "'", "''", "``", "…", "۔۔۔"
}

STRIP_CHARS = "۔،!?؟؛:,.()[]{}\"'—–…"


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

# -----------------------------
# POS resources (expanded Urdu-aware rules, mapped to assignment tagset)
# Final output tags remain:
# NOUN VERB ADJ ADV PRON DET CONJ POST NUM PUNC UNK
# -----------------------------
MONTH_NAMES = {
    "جنوری", "فروری", "مارچ", "اپریل", "مئی", "جون",
    "جولائی", "اگست", "ستمبر", "اکتوبر", "نومبر", "دسمبر"
}

DAY_NAMES = {
    "پیر", "منگل", "بدھ", "جمعرات", "جمعہ", "ہفتہ", "اتوار"
}

# Pronouns: personal, demonstrative, relative, reflexive-ish forms
PRONOUNS = {
    "میں", "ہم", "تم", "آپ", "وہ", "یہ", "اس", "ان",
    "جس", "جن", "جو", "جسے", "جنہیں", "انہیں", "انھیں",
    "مجھے", "ہمیں", "تمہیں", "اپ", "اپنا", "اپنی", "اپنے",
    "خود"
}

# Determiners / quantifiers / articles-like forms
DETERMINERS = {
    "ایک", "ہر", "کچھ", "تمام", "کوئی", "کسی", "چند",
    "اسی", "ایسی", "ایسا", "ایسے", "دوسرا", "دوسری", "دوسرے",
    "پہلا", "پہلی", "پہلے", "متعدد", "بعض"
}

# Conjunctions
CONJUNCTIONS = {
    "اور", "یا", "لیکن", "بلکہ", "اگر", "مگر", "تاہم",
    "جبکہ", "کہ", "کیونکہ", "تو", "چنانچہ"
}

# Postpositions
POSTPOSITIONS = {
    "میں", "پر", "سے", "کو", "کا", "کی", "کے", "تک", "بعد",
    "ساتھ", "لئے", "لیے", "تحت", "بارے", "طور", "پاس",
    "جانب", "خلاف", "بھی", "بھر", "اندر", "باہر", "اوپر", "نیچے"
}

# Adverbs: time, degree, frequency, place, negation
ADVERBS = {
    "اب", "پھر", "زیادہ", "کم", "بہت", "جلد", "اکثر", "ہمیشہ",
    "کبھی", "شاید", "تقریبا", "تقریباً", "مزید", "یہاں", "وہاں",
    "آج", "کل", "فوراً", "دوبارہ", "نہ", "نہیں"
}

# Core verb list: auxiliaries, copulas, light verbs, common finite forms
VERBS = {
    "ہے", "ہیں", "تھا", "تھی", "تھے",
    "ہو", "ہوا", "ہوئی", "ہوئے",
    "کر", "کیا", "کی", "کئے", "کیے", "کرے",
    "کرتا", "کرتی", "کرتے",
    "جا", "جاتا", "جاتی", "جاتے",
    "گیا", "گئی", "گئے",
    "رہا", "رہی", "رہے",
    "لگا", "لگی", "لگے",
    "لیا", "لی", "لیں", "لے",
    "دیا", "دی", "دیے",
    "ملا", "ملی", "ملے",
    "رکھا", "رکھی", "رکھے",
    "چاہ", "چاہا", "چاہے", "چاہتی", "چاہتے",
    "سکتا", "سکتی", "سکتے",
    "چاہیے", "پڑا", "پڑی", "پڑے",
    "بتایا", "کہا", "سنا", "پہنچا", "پہنچی", "پہنچے",
    "شروع", "ختم", "ہلاک", "شامل"
}

# Adjectives: descriptive, comparative, evaluative
ADJECTIVES = {
    "بڑا", "بڑی", "بڑے",
    "چھوٹا", "چھوٹی", "چھوٹے",
    "اہم", "قومی", "سیاسی", "مقامی", "عالمی",
    "مختلف", "سابق", "شدید", "خوبصورت", "سخت", "ضروری", "کامیاب",
    "واضح", "حالیہ", "متعلقہ", "مبینہ", "بہتر", "کمتر",
    "اچھا", "اچھی", "اچھے", "پہلا", "پہلی", "پہلے",
    "دوسرا", "دوسری", "دوسرے", "غیر", "طویل"
}

# Some content words that are often mistaken because of noisy corpus;
# keep them as nouns by default
COMMON_NOUN_HINTS = {
    "سال", "ماہ", "دن", "رات", "وقت", "خبر", "رپورٹ", "حکومت", "عدالت",
    "فوج", "ملک", "شہر", "خاندان", "جنگ", "موقع", "روزگار", "تعلیم",
    "صحت", "پولیس", "وزیر", "صدر", "وزیراعظم", "ڈاکٹر", "ہسپتال",
    "میچ", "ٹیم", "کرکٹ", "پاکستان", "انڈیا", "امریکہ", "ایران", "روس", "چین"
}

# Optional grouped lookup
POS_LEXICON = {
    "PRON": PRONOUNS,
    "DET": DETERMINERS,
    "CONJ": CONJUNCTIONS,
    "POST": POSTPOSITIONS,
    "ADV": ADVERBS,
    "VERB": VERBS,
    "ADJ": ADJECTIVES,
}


def normalize_token_for_rules(tok: str) -> str:
    return tok.strip(STRIP_CHARS).strip()


def is_number_token(tok: str) -> bool:
    return tok.isdigit() or tok == "<NUM>"


def is_punctuation_token(tok: str) -> bool:
    if tok in PUNCT_TOKENS:
        return True
    stripped = normalize_token_for_rules(tok)
    return stripped == ""


def pos_tag_token(tok: str) -> str:
    """
    Rich Urdu-aware rule-based POS tagging mapped into the assignment's reduced tagset.
    """

    # 1) Punctuation / empty
    if is_punctuation_token(tok):
        return "PUNC"

    # 2) Numbers
    if is_number_token(tok):
        return "NUM"

    clean_tok = normalize_token_for_rules(tok)

    if clean_tok == "":
        return "PUNC"

    if is_number_token(clean_tok):
        return "NUM"

    # 3) Calendar words
    if clean_tok in MONTH_NAMES or clean_tok in DAY_NAMES:
        return "NOUN"

    # 4) High-priority exact lexical mappings
    # Put POST before PRON because forms like "میں" are more useful as POST
    if clean_tok in POSTPOSITIONS:
        return "POST"
    if clean_tok in CONJUNCTIONS:
        return "CONJ"
    if clean_tok in ADVERBS:
        return "ADV"
    if clean_tok in DETERMINERS:
        return "DET"
    if clean_tok in PRONOUNS:
        return "PRON"
    if clean_tok in VERBS:
        return "VERB"
    if clean_tok in ADJECTIVES:
        return "ADJ"

    # 5) Relative/interrogative pronoun-like forms
    if clean_tok in {"جس", "جن", "جو", "جسے", "جنہیں", "کس", "کون"}:
        return "PRON"

    # 6) Negation particles treated as adverbs in reduced tagset
    if clean_tok in {"نہ", "نہیں"}:
        return "ADV"

    # 7) Common infinitive / verbal endings
    if clean_tok.endswith(("نا", "نے")):
        return "VERB"

    # 8) Common finite/present/past participial endings
    if clean_tok.endswith(("تا", "تی", "تے", "گا", "گی", "گے")):
        return "VERB"

    # 9) Common adjective-like endings
    if clean_tok.endswith(("انہ", "وار", "مند", "دار")) and len(clean_tok) > 3:
        return "ADJ"

    # 10) Comparative/superlative-like forms
    if clean_tok in {"بہتر", "کمتر"}:
        return "ADJ"

    # 11) Common noun hints
    if clean_tok in COMMON_NOUN_HINTS:
        return "NOUN"

    # 12) Very short noisy fragments
    if len(clean_tok) == 1:
        return "UNK"

    # 13) Default content-word fallback
    return "NOUN"


def pos_tag_sentence(tokens: List[str]) -> List[str]:
    return [pos_tag_token(tok) for tok in tokens]

# -----------------------------
# NER tagging
# -----------------------------
def tag_span(tags: List[str], start: int, end: int, entity_type: str) -> None:
    if start >= end:
        return
    tags[start] = f"B-{entity_type}"
    for i in range(start + 1, end):
        tags[i] = f"I-{entity_type}"


def try_match_multi(tokens: List[str], i: int, patterns: set) -> int:
    for pattern in sorted(patterns, key=len, reverse=True):
        n = len(pattern)
        if i + n <= len(tokens) and tuple(tokens[i:i+n]) == pattern:
            return n
    return 0


def ner_tag_sentence(tokens: List[str]) -> List[str]:
    tags = ["O"] * len(tokens)
    i = 0

    while i < len(tokens):
        # multi-token PERSON
        m = try_match_multi(tokens, i, PERSON_MULTI)
        if m > 0:
            tag_span(tags, i, i + m, "PER")
            i += m
            continue

        # multi-token LOCATION
        m = try_match_multi(tokens, i, LOCATION_MULTI)
        if m > 0:
            tag_span(tags, i, i + m, "LOC")
            i += m
            continue

        # multi-token ORG
        m = try_match_multi(tokens, i, ORG_MULTI)
        if m > 0:
            tag_span(tags, i, i + m, "ORG")
            i += m
            continue

        tok = normalize_token_for_rules(tokens[i])

        if tok in PERSON_GAZETTEER:
            tags[i] = "B-PER"
            i += 1
            continue

        if tok in LOCATION_GAZETTEER:
            tags[i] = "B-LOC"
            i += 1
            continue

        if tok in ORG_GAZETTEER:
            tags[i] = "B-ORG"
            i += 1
            continue

        if tok in {"ورلڈ", "کپ", "بجٹ", "ویکسین", "ٹیکسٹائل"}:
            tags[i] = "B-MISC"

        i += 1

    return tags


# -----------------------------
# Annotation pipeline
# -----------------------------
def annotate_examples(selected_df: pd.DataFrame) -> List[Dict[str, Any]]:
    annotated_examples: List[Dict[str, Any]] = []

    for idx, row in selected_df.iterrows():
        tokens = row["tokens"]
        pos_tags = pos_tag_sentence(tokens)
        ner_tags = ner_tag_sentence(tokens)

        annotated_examples.append({
            "sentence_id": int(idx),
            "article_id": row["article_id"],
            "title": row["title"],
            "topic": row["topic"],
            "tokens": tokens,
            "pos_tags": pos_tags,
            "ner_tags": ner_tags
        })

    return annotated_examples


def stratified_split(
    annotated_examples: List[Dict[str, Any]],
    random_state: int = SEED
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    example_df = pd.DataFrame([
        {"idx": i, "topic": ex["topic"]}
        for i, ex in enumerate(annotated_examples)
    ])

    train_idx, temp_idx = train_test_split(
        example_df["idx"],
        test_size=0.30,
        random_state=random_state,
        stratify=example_df["topic"]
    )

    temp_df = example_df.iloc[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_df["idx"],
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df["topic"]
    )

    train_data = [annotated_examples[i] for i in train_idx]
    val_data = [annotated_examples[i] for i in val_idx]
    test_data = [annotated_examples[i] for i in test_idx]

    return train_data, val_data, test_data


def label_distribution(split_data: List[Dict[str, Any]], task: str) -> pd.Series:
    labels: List[str] = []
    key = "pos_tags" if task == "pos" else "ner_tags"
    for ex in split_data:
        labels.extend(ex[key])
    return pd.Series(labels).value_counts()


def topic_distribution(split_data: List[Dict[str, Any]]) -> pd.Series:
    return pd.Series([ex["topic"] for ex in split_data]).value_counts()


def write_conll(split_data: List[Dict[str, Any]], filepath: str, task: str = "pos") -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for ex in split_data:
            tokens = ex["tokens"]
            labels = ex["pos_tags"] if task == "pos" else ex["ner_tags"]

            for tok, lab in zip(tokens, labels):
                f.write(f"{tok}\t{lab}\n")
            f.write("\n")


def build_tag_mappings() -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    pos2idx = {tag: i for i, tag in enumerate(POS_TAGS)}
    idx2pos = {i: tag for tag, i in pos2idx.items()}

    ner2idx = {tag: i for i, tag in enumerate(NER_TAGS)}
    idx2ner = {i: tag for tag, i in ner2idx.items()}

    return pos2idx, idx2pos, ner2idx, idx2ner


def prepare_part2_data(
    cleaned_path: str,
    metadata_path: str,
    data_dir: str,
    target_size: int = 500
) -> Dict[str, Any]:
    os.makedirs(data_dir, exist_ok=True)

    articles = load_articles(cleaned_path, metadata_path)
    sentence_df = build_sentence_pool(articles)
    selected_df = select_balanced_sentences(sentence_df, target_size=target_size)
    annotated_examples = annotate_examples(selected_df)

    train_data, val_data, test_data = stratified_split(annotated_examples)
    pos2idx, idx2pos, ner2idx, idx2ner = build_tag_mappings()

    write_conll(train_data, os.path.join(data_dir, "pos_train.conll"), task="pos")
    write_conll(test_data, os.path.join(data_dir, "pos_test.conll"), task="pos")
    write_conll(train_data, os.path.join(data_dir, "ner_train.conll"), task="ner")
    write_conll(test_data, os.path.join(data_dir, "ner_test.conll"), task="ner")

    return {
        "articles": articles,
        "sentence_df": sentence_df,
        "selected_df": selected_df,
        "annotated_examples": annotated_examples,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "pos2idx": pos2idx,
        "idx2pos": idx2pos,
        "ner2idx": ner2idx,
        "idx2ner": idx2ner,
    }