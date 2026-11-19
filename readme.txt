# Domain-Specific Document Classifier (Finance)

A domain-specific NLP classification and summarization system for **Finance**, built with a fine-tuned **Transformer** (`distilbert-base-uncased`) across **20 financial categories**, an extractive **TF-IDF** document summarizer, and an interactive **Streamlit** GUI.

## Overview

Financial documents—such as earnings updates, central bank statements, M&A announcements, analyst notes, and market reports—require domain-aware text processing and accurate categorization. This project cleans and prepares financial text while preserving numbers, currency symbols, and percentages, benchmarks a classical baseline, fine-tunes a Transformer classifier on `zeroshot/twitter-financial-news-topic`, evaluates classification performance, and serves predictions with confidence scores and extractive summaries in Streamlit.

## Repository Structure

```text
financial-document-classifier/
├── app.py                      # Streamlit web GUI
├── src/
│   ├── __init__.py
│   ├── preprocess.py               # Domain text cleaning & TF-IDF extractive summarizer
│   └── train_classifier.py         # Data split, baseline, Transformer fine-tuning & evaluation
├── data/
│   └── sample_financial_docs.txt   # Sample test documents for the GUI
├── models/
│   ├── fin_doc_classifier/         # Fine-tuned Transformer model & tokenizer
│   └── confusion_matrix.png        # Test set confusion matrix plot
├── screenshots/                # Working Streamlit GUI screenshots
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md


Setup & Installation

Clone the repository:



Bash

git clone https://github.com/haru-6/financial-document-classifier.git
cd financial-document-classifier
Create and activate a virtual environment (Python 3.12 recommended):


Bash

py -3.12 -m venv .venv312
.venv312\Scripts\Activate.ps1
Install dependencies:


Bash

pip install --upgrade pip
pip install -r requirements.txt


Usage

1. Train & Evaluate the Classifier
Run the training pipeline to clean/deduplicate data, train a TF-IDF + Logistic Regression baseline, fine-tune distilbert-base-uncased on the 20 financial topics, and evaluate on the held-out test set (Accuracy, Precision, Recall, F1-score, and Confusion Matrix):

Bash

python -m src.train_classifier



2. Launch the Streamlit Web App

Bash

streamlit run app.py


In the web interface, you can:

Paste financial text directly or upload .txt, .pdf, or .docx documents.

Click Classify Document to view the predicted financial category, confidence score, top category probabilities, and an extractive document summary.




Screenshots

Classification Result & Confidence Score: screenshots/classification_result.png

Document Summary View: screenshots/0document_summary.png
