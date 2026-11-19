import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.preprocess import clean_financial_text

# 20 financial topic categories in zeroshot/twitter-financial-news-topic
TOPIC_LABELS = {
    0: "Analyst Update",
    1: "Fed | Central Banks",
    2: "Company | Product News",
    3: "Treasuries | Corporate Debt",
    4: "Dividend",
    5: "Earnings",
    6: "Energy | Oil",
    7: "Financials",
    8: "Currencies",
    9: "General News | Opinion",
    10: "Gold | Metals | Materials",
    11: "IPO",
    12: "Legal | Regulation",
    13: "M&A | Investments",
    14: "Macro",
    15: "Markets",
    16: "Politics",
    17: "Personnel Change",
    18: "Stock Commentary",
    19: "Stock Movement",
}

MODEL_NAME = "distilbert-base-uncased"
SAVE_DIR = "models/fin_doc_classifier"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def main() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # Stage 1 & 2: Load Domain-Specific Labeled Finance Dataset
    print("Loading zeroshot/twitter-financial-news-topic...")
    raw_ds = load_dataset("zeroshot/twitter-financial-news-topic")
    df_all = pd.concat(
        [raw_ds["train"].to_pandas(), raw_ds["validation"].to_pandas()],
        ignore_index=True,
    )

    # Stage 3: Clean & Deduplicate
    df_all = df_all.dropna(subset=["text", "label"]).copy()
    df_all["clean_text"] = df_all["text"].apply(clean_financial_text)
    df_all = df_all[df_all["clean_text"].str.len() >= 10]
    df_all = df_all.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    df_all["label"] = df_all["label"].astype(int)

    # Stage 5: Quick EDA Summary
    print(f"Total clean unique financial documents: {len(df_all)}")
    print(f"Number of categories: {df_all['label'].nunique()}")
    word_counts = df_all["clean_text"].str.split().str.len()
    print(f"Word count -> Mean: {word_counts.mean():.1f} | 95th percentile: {np.percentile(word_counts, 95):.0f}")

    # Stratified Train / Validation / Test Split (80% / 10% / 10%)
    train_df, temp_df = train_test_split(
        df_all, test_size=0.20, stratify=df_all["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    # Save a sample multi-paragraph finance text file for testing the GUI
    sample_texts = test_df["clean_text"].head(5).tolist()
    with open("data/sample_financial_docs.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(sample_texts))

    # Stage 6A: Classical ML Baseline (TF-IDF + Logistic Regression)
    print("\nTraining TF-IDF + Logistic Regression baseline...")
    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(train_df["clean_text"])
    X_test_tfidf = tfidf.transform(test_df["clean_text"])

    baseline_clf = LogisticRegression(max_iter=1000)
    baseline_clf.fit(X_train_tfidf, train_df["label"])
    baseline_preds = baseline_clf.predict(X_test_tfidf)
    baseline_acc = accuracy_score(test_df["label"], baseline_preds)
    _, _, baseline_f1, _ = precision_recall_fscore_support(
        test_df["label"], baseline_preds, average="macro", zero_division=0
    )
    print(f"Baseline Test Accuracy: {baseline_acc:.4f} | Macro F1: {baseline_f1:.4f}")

    # Stage 4 & 6B: Transformer Subword Tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_batch(batch):
        return tokenizer(batch["clean_text"], truncation=True, max_length=128)

    train_hf = Dataset.from_pandas(train_df[["clean_text", "label"]], preserve_index=False).map(
        tokenize_batch, batched=True
    )
    val_hf = Dataset.from_pandas(val_df[["clean_text", "label"]], preserve_index=False).map(
        tokenize_batch, batched=True
    )
    test_hf = Dataset.from_pandas(test_df[["clean_text", "label"]], preserve_index=False).map(
        tokenize_batch, batched=True
    )

    # Stage 7: Fine-Tune Transformer Classifier
    id2label = {k: v for k, v in TOPIC_LABELS.items()}
    label2id = {v: k for k, v in TOPIC_LABELS.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(TOPIC_LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    # Use smaller batch size if running on CPU
    batch_size = 32 if device == "cuda" else 16

    training_args = TrainingArguments(
        output_dir="./trainer_checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"\nFine-tuning {MODEL_NAME}...")
    trainer.train()

    # Stage 8: Evaluate on Held-Out Test Set
    print("\nEvaluating fine-tuned model on test set...")
    test_output = trainer.predict(test_hf)
    test_preds = np.argmax(test_output.predictions, axis=-1)
    test_labels = test_df["label"].to_numpy()

    print("\nTest Metrics:", test_output.metrics)
    target_names = [TOPIC_LABELS[i] for i in range(len(TOPIC_LABELS))]
    print("\nClassification Report:\n")
    print(classification_report(test_labels, test_preds, target_names=target_names, zero_division=0))

    # Plot & Save Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("True Category")
    ax.set_title("Financial Document Classifier - Test Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150)

    # Save Final Fine-Tuned Model & Tokenizer for Streamlit GUI
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"\nSaved fine-tuned Transformer model and tokenizer to ./{SAVE_DIR}/")


if __name__ == "__main__":
    main()