# NLP Assignment 2 — Neural NLP Pipeline

**GitHub:** Sameer-Khan08

This repository contains a from-scratch PyTorch implementation of three major components for the BBC Urdu NLP pipeline:

1. **Part 1 — Word Embeddings**
   - TF-IDF weighted document representation
   - PPMI weighted co-occurrence representation
   - Skip-gram Word2Vec with negative sampling
   - Four-condition comparison (C1–C4)

2. **Part 2 — Sequence Labeling**
   - Rule-based POS annotation
   - Gazetteer-based BIO NER annotation
   - 2-layer BiLSTM POS tagger
   - 2-layer BiLSTM NER tagger with and without CRF
   - Frozen vs fine-tuned embeddings
   - Ablation study (A1–A4)

3. **Part 3 — Topic Classification**
   - Transformer encoder classifier implemented from scratch
   - BiLSTM topic-classification baseline
   - BiLSTM vs Transformer comparison

---

## Repository Structure

```text
i23-XXXX_Assignment2_DS-X/
│
├── i23-XXXX_Assignment2_DS-X.ipynb
├── report.pdf
├── README.md
│
├── embeddings/
│   ├── tfidf_matrix.npy
│   ├── ppmi_matrix.npy
│   ├── embeddings_w2v.npy
│   └── word2idx.json
│
├── models/
│   ├── bilstm_pos.pt
│   ├── bilstm_ner.pt
│   └── transformer_cls.pt
│
├── data/
│   ├── pos_train.conll
│   ├── pos_test.conll
│   ├── ner_train.conll
│   └── ner_test.conll
│
├── Sequence Labeling POS Tagging & NER/
│   ├── Sequence Labeling.ipynb
│   └── src/
│       ├── __init__.py
│       ├── crf.py
│       ├── data_and_annotation.py
│       ├── datasets.py
│       ├── models.py
│       └── train_eval.py
│
└── Transformer Encoder for Topic Classification/
    ├── Transformer Encoder for Topic Classification.ipynb
    └── src/
        ├── __init__.py
        ├── bilstm_topic.py
        ├── data_topic.py
        ├── train_eval_topic.py
        └── transformer_model.py
```

---

## Requirements

Recommended environment:

- Python 3.10+
- PyTorch
- NumPy
- pandas
- scikit-learn
- matplotlib
- tqdm

Install the main Python packages:

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm
```

---

## Required Input Files

Place these files in the working directory before running the notebooks:

- `cleaned.txt`
- `raw.txt`
- `metadata.json`

These are used as follows:

- `cleaned.txt` → Parts 1, 2, and 3
- `raw.txt` → Part 1 comparison and Part 2 baseline-related steps
- `metadata.json` → topic/category assignment and document metadata

---

## Part 1 — Word Embeddings

Notebook:
- `Part1.ipynb`

### What Part 1 does
- Builds a term-document matrix from `cleaned.txt`
- Computes TF-IDF and saves:
  - `tfidf_matrix.npy`
- Builds a co-occurrence matrix with context window `k = 5`
- Computes PPMI and saves:
  - `ppmi_matrix.npy`
- Trains Skip-gram Word2Vec and saves:
  - `embeddings_w2v.npy`
  - `word2idx.json`
- Evaluates nearest neighbours, analogies, and four conditions:
  - `C1`: PPMI baseline
  - `C2`: Skip-gram on `raw.txt`
  - `C3`: Skip-gram on `cleaned.txt`
  - `C4`: Skip-gram on `cleaned.txt` with `d = 200`

### How to run
Open and execute all cells in:

```text
Part1.ipynb
```

### Important outputs
- TF-IDF matrix
- PPMI matrix
- Word2Vec embeddings
- Loss curves
- t-SNE plot
- MRR comparison table

---

## Part 2 — Sequence Labeling

Notebook:
- `Sequence Labeling POS Tagging & NER/Sequence Labeling.ipynb`

### What Part 2 does
- Selects 500 sentences from the cleaned corpus
- Builds weakly supervised POS labels
- Builds weakly supervised BIO NER labels
- Splits the dataset 70/15/15
- Saves:
  - `data/pos_train.conll`
  - `data/pos_test.conll`
  - `data/ner_train.conll`
  - `data/ner_test.conll`
- Trains:
  - POS BiLSTM with frozen embeddings
  - POS BiLSTM with fine-tuned embeddings
  - NER BiLSTM + Softmax
  - NER BiLSTM + CRF
- Runs ablations:
  - `A1`: Unidirectional LSTM only
  - `A2`: No dropout
  - `A3`: Random embedding initialization
  - `A4`: Softmax instead of CRF

### Source files
Located in:

```text
Sequence Labeling POS Tagging & NER/src/
```

Main modules:
- `data_and_annotation.py` → data loading, topic assignment, POS/NER weak annotation
- `datasets.py` → dataset and dataloader preparation
- `models.py` → BiLSTM models and CRF-integrated model
- `crf.py` → linear-chain CRF + Viterbi decoding
- `train_eval.py` → training loops, evaluation, confusion analysis

### How to run
1. Ensure `word2idx.json` and `embeddings_w2v.npy` from Part 1 are available.
2. Open and run:

```text
Sequence Labeling POS Tagging & NER/Sequence Labeling.ipynb
```

### Notes
- POS results were much stronger than NER.
- NER labels were generated with weak supervision, so entity-level performance remained limited.
- The CRF layer did not significantly improve NER because the model largely collapsed to predicting the majority class.

---

## Part 3 — Topic Classification

Notebook:
- `Transformer Encoder for Topic Classification/Transformer Encoder for Topic Classification.ipynb`

### What Part 3 does
- Assigns each article to one of 5 categories:
  - Politics
  - Sports
  - Economy
  - International
  - Health & Society
- Builds padded token-ID document sequences
- Splits the dataset 70/15/15
- Implements from scratch:
  - Scaled dot-product attention
  - Multi-head self-attention
  - Position-wise feed-forward network
  - Transformer encoder classifier with `[CLS]`
- Trains a BiLSTM topic-classification baseline
- Compares BiLSTM vs Transformer

### Source files
Located in:

```text
Transformer Encoder for Topic Classification/src/
```

Main modules:
- `data_topic.py` → topic dataset creation and stratified split
- `transformer_model.py` → full Transformer encoder implementation
- `train_eval_topic.py` → training/evaluation utilities
- `bilstm_topic.py` → BiLSTM topic-classification baseline

### How to run
Open and execute:

```text
Transformer Encoder for Topic Classification/Transformer Encoder for Topic Classification.ipynb
```

### Notes
- The BiLSTM topic classifier achieved slightly better results than the Transformer on the small dataset.
- Both models peaked very early, indicating limited data size.
- Attention heatmaps were generated for correctly classified articles using the final encoder layer.

---

## Saved Model Files

Suggested saved model outputs:

- `models/bilstm_pos.pt`
- `models/bilstm_ner.pt`
- `models/transformer_cls.pt`

If these are not yet saved in your local copy, you can save them in the notebooks using:

```python
torch.save(model.state_dict(), "models/model_name.pt")
```

---

## Reproducibility Notes

- The code uses fixed random seeds where possible.
- If imports behave unexpectedly after moving files, restart the Jupyter kernel and rerun cells from the top.
- For package-based imports, all project modules should be imported through `src`.

Example:

```python
from src.models import POSTagger
from src.train_eval import train_sequence_tagger
```

---

## Key Findings

### Part 1
- The cleaned corpus produced better embeddings than the raw corpus.
- Increasing embedding size did not automatically guarantee better performance.
- MRR values remained modest because of token noise and limited corpus size.

### Part 2
- POS tagging performed well, especially with fine-tuned embeddings.
- NER performance remained weak due to sparse and noisy weakly supervised labels.
- The CRF layer did not help much because the emission quality was poor.

### Part 3
- The BiLSTM topic classifier slightly outperformed the Transformer encoder.
- On a very small dataset, the BiLSTM appears to be the more appropriate architecture.
- Transformer attention maps still provided useful interpretability.

---

## How to Reproduce the Full Assignment

### Step 1
Run `Part1.ipynb` to generate:

- `tfidf_matrix.npy`
- `ppmi_matrix.npy`
- `embeddings_w2v.npy`
- `word2idx.json`

### Step 2
Run `Sequence Labeling POS Tagging & NER/Sequence Labeling.ipynb` to generate:

- `.conll` sequence-labeling files
- POS and NER models
- POS/NER evaluation tables and plots

### Step 3
Run `Transformer Encoder for Topic Classification/Transformer Encoder for Topic Classification.ipynb` to generate:

- Transformer topic-classification results
- BiLSTM topic baseline results
- BiLSTM vs Transformer comparison
- Attention heatmaps

---

## GitHub

GitHub username used for this project:

**Sameer-Khan08**

If publishing the repository, use the assignment-required public repository naming convention from the brief.

---

## Final Note

All implementations were done in PyTorch from scratch, following the assignment restriction of not using built-in Transformer modules such as:

- `nn.Transformer`
- `nn.MultiheadAttention`
- `nn.TransformerEncoder`

