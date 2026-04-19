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

PUNCT_TOKENS = {"۔", "،", ".", ",", "!", "؟", ":", ";", "(", ")", "[", "]", "{", "}", "-", "—", '"'}


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


POS_LEXICON = {
    "PRON": {
        "میں", "ہم", "تم", "آپ", "وہ", "یہ", "ان", "اس", "انہیں", "انھیں",
        "مجھے", "ہمیں", "تمہیں", "اپ", "اپنے", "اپنی", "اپنا"
    },
    "DET": {
        "ایک", "یہ", "وہ", "اس", "ان", "کچھ", "تمام", "ہر", "کسی", "کوئی"
    },
    "CONJ": {
        "اور", "یا", "لیکن", "بلکہ", "اگر", "مگر", "تاہم", "جبکہ", "کہ"
    },
    "POST": {
        "میں", "پر", "سے", "کو", "کا", "کی", "کے", "تک", "بعد",
        "ساتھ", "لئے", "لیے", "تحت", "بارے", "طور", "پاس"
    },
    "ADV": {
        "اب", "پھر", "زیادہ", "کم", "بہت", "جلد", "اکثر", "ہمیشہ", "کبھی",
        "شاید", "تقریبا", "تقریباً", "مزید"
    },
    "VERB": {
        "ہے", "ہیں", "تھا", "تھی", "تھے", "ہو", "ہوا", "ہوئی", "ہوئے",
        "کر", "کیا", "کی", "کئے", "کرے", "کرتا", "کرتی", "کرتے",
        "گیا", "گئی", "گئے", "جا", "جاتا", "جاتی", "جاتے",
        "رہا", "رہی", "رہے", "لگا", "لگی", "لگے",
        "دیا", "دی", "دیے", "لیا", "لی", "لیں", "لے", "ملا", "ملی", "ملے"
    },
    "ADJ": {
        "بڑا", "بڑی", "بڑے", "اہم", "قومی", "سیاسی", "مقامی", "عالمی",
        "مختلف", "سابق", "شدید", "خوبصورت", "سخت", "ضروری", "کامیاب"
    }
}


PERSON_GAZETTEER = {
    "عمران", "نواز", "شہباز", "مریم", "زرداری", "بشری", "اسحاق", "محسن",
    "عاصم", "فیض", "شاہد", "رضا", "احمد", "حسین", "علی", "حمید", "سرفراز",
    "صبا", "قمر", "طاہر", "خالد", "فاطمہ", "شازیہ", "رحمان", "عثمان",
    "شہزاد", "اکبر", "عارف", "قریشی", "محمد", "ریاض", "احسن", "کاشف",
    "حسن", "طفیل", "مرزا", "ڈار", "منیر", "اوپیندر", "پوتن", "ٹرمپ",
    "نتن", "یاہو", "اردوغان", "نور", "ناصر", "مجوکہ", "کنڈی", "فیصل"
}

LOCATION_GAZETTEER = {
    "پاکستان", "پنجاب", "سندھ", "بلوچستان", "خیبر", "پختونخوا", "اسلام", "اباد",
    "لاہور", "کراچی", "راولپنڈی", "پشاور", "کوئٹہ", "مری",
    "شیخوپورہ", "حیدرآباد", "سوات", "ہنزہ", "چترال", "کشمیر",
    "غزہ", "ایران", "امریکہ", "روس", "چین", "بھارت", "انڈیا", "ترکی",
    "عمان", "افغانستان", "ترکمانستان", "ڈیووس", "لندن",
    "سیالکوٹ", "پسرور", "کرم", "چپورسن", "نوشکی", "قلات",
    "ٹانک", "جنڈولہ", "صدہ", "مکہ", "مدینہ", "اشک", "آباد",
    "فیصل", "آباد"
}

ORG_GAZETTEER = {
    "پی", "آئی", "اے", "نیپرا", "سپارکو", "آئی", "ایس", "سی", "ٹی",
    "ڈی", "عدالت", "سپریم", "کورٹ", "اقوام", "متحدہ", "وزارت", "خارجہ",
    "پولیس", "حکومت", "پارلیمان", "اسمبلی", "فوج", "یونیورسٹی",
    "ہسپتال", "ایگزیوس", "بی", "سی"
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


def load_articles(cleaned_path: str, metadata_path: str) -> List[Dict[str, Any]]:
    cleaned_text = read_text(cleaned_path)
    metadata = read_json(metadata_path)

    cleaned_blocks = parse_numbered_blocks(cleaned_text)
    common_ids = sorted(
        set(cleaned_blocks.keys()) & set(metadata.keys()),
        key=lambda x: int(x)
    )

    articles: List[Dict[str, Any]] = []
    for doc_id in common_ids:
        cleaned_sentences = extract_cleaned_sentences(cleaned_blocks[doc_id])
        cleaned_tokens = [tok for sent in cleaned_sentences for tok in sent]

        article = {
            "id": doc_id,
            "title": metadata[doc_id].get("title", ""),
            "publish_date": metadata[doc_id].get("publish_date", ""),
            "cleaned_sentences": cleaned_sentences,
            "cleaned_tokens": cleaned_tokens,
        }
        article["topic"] = assign_topic(article["title"], article["cleaned_tokens"])
        articles.append(article)

    return articles


def build_sentence_pool(articles: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for article in articles:
        for sent in article["cleaned_sentences"]:
            if 4 <= len(sent) <= 40:
                rows.append({
                    "article_id": article["id"],
                    "title": article["title"],
                    "topic": article["topic"],
                    "tokens": sent
                })
    return pd.DataFrame(rows)


def select_balanced_sentences(
    sentence_df: pd.DataFrame,
    target_size: int = 500,
    min_per_topic: int = 100,
    core_topics: List[str] = None,
    random_state: int = SEED
) -> pd.DataFrame:
    if core_topics is None:
        core_topics = ["politics", "sports", "economy"]

    selected_rows = []
    for topic in core_topics:
        topic_df = sentence_df[sentence_df["topic"] == topic]
        n_take = min(min_per_topic, len(topic_df))
        if n_take == 0:
            continue
        selected_rows.append(topic_df.sample(n=n_take, random_state=random_state))

    if selected_rows:
        selected_df = pd.concat(selected_rows, ignore_index=False)

        # create a hashable key for duplicate removal
        selected_df = selected_df.copy()
        selected_df["_tokens_key"] = selected_df["tokens"].apply(lambda x: tuple(x))
        selected_df = selected_df.drop_duplicates(subset=["article_id", "topic", "_tokens_key"])
        selected_df = selected_df.drop(columns=["_tokens_key"])
    else:
        selected_df = pd.DataFrame(columns=sentence_df.columns)

    remaining_needed = target_size - len(selected_df)
    remaining_pool = sentence_df.drop(selected_df.index, errors="ignore")

    if remaining_needed > 0 and len(remaining_pool) > 0:
        extra_rows = remaining_pool.sample(
            n=min(remaining_needed, len(remaining_pool)),
            random_state=random_state
        )
        selected_df = pd.concat([selected_df, extra_rows], ignore_index=False)

        selected_df = selected_df.copy()
        selected_df["_tokens_key"] = selected_df["tokens"].apply(lambda x: tuple(x))
        selected_df = selected_df.drop_duplicates(subset=["article_id", "topic", "_tokens_key"])
        selected_df = selected_df.drop(columns=["_tokens_key"])

    selected_df = selected_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return selected_df


def is_number_token(tok: str) -> bool:
    return tok.isdigit() or tok == "<NUM>"


def pos_tag_token(tok: str) -> str:
    if tok in PUNCT_TOKENS:
        return "PUNC"
    if is_number_token(tok):
        return "NUM"

    for tag in ["PRON", "DET", "CONJ", "POST", "ADV", "VERB", "ADJ"]:
        if tok in POS_LEXICON[tag]:
            return tag

    if tok.endswith("گی") or tok.endswith("گا") or tok.endswith("گے"):
        return "VERB"
    if tok.endswith("انہ") or tok.endswith("ی"):
        return "ADJ"
    if len(tok) >= 2 and tok not in {"ہے", "ہیں", "تھا", "تھی", "تھے"}:
        return "NOUN"

    return "UNK"


def pos_tag_sentence(tokens: List[str]) -> List[str]:
    return [pos_tag_token(tok) for tok in tokens]


def ner_tag_sentence(tokens: List[str]) -> List[str]:
    tags = ["O"] * len(tokens)
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        if tok in PERSON_GAZETTEER:
            tags[i] = "B-PER"
            j = i + 1
            while j < len(tokens) and tokens[j] in PERSON_GAZETTEER:
                tags[j] = "I-PER"
                j += 1
            i = j
            continue

        if tok in LOCATION_GAZETTEER:
            tags[i] = "B-LOC"
            j = i + 1
            while j < len(tokens) and tokens[j] in LOCATION_GAZETTEER:
                tags[j] = "I-LOC"
                j += 1
            i = j
            continue

        if tok in ORG_GAZETTEER:
            tags[i] = "B-ORG"
            j = i + 1
            while j < len(tokens) and tokens[j] in ORG_GAZETTEER:
                tags[j] = "I-ORG"
                j += 1
            i = j
            continue

        if tok in {"ورلڈ", "کپ", "بجٹ", "ویکسین", "ٹیکسٹائل"}:
            tags[i] = "B-MISC"

        i += 1

    return tags


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