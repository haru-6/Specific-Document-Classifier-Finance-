import docx
import matplotlib.pyplot as plt
import pandas as pd
import pypdf
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.preprocess import clean_financial_text, extractive_summary

st.set_page_config(
    page_title="Financial Document Classifier",
    page_icon="🏦",
    layout="wide",
)

MODEL_DIR = "models/fin_doc_classifier"


@st.cache_resource
def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def extract_text_from_upload(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        reader = pypdf.PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    if name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return ""


def predict_document(text: str, tokenizer, model) -> tuple[str, float, pd.DataFrame]:
    cleaned = clean_financial_text(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()

    id2label = model.config.id2label
    records = [
        {"Category": id2label[i], "Probability": float(probs[i])}
        for i in range(len(probs))
    ]
    prob_df = pd.DataFrame(records).sort_values("Probability", ascending=False).reset_index(drop=True)
    top_category = prob_df.loc[0, "Category"]
    top_confidence = prob_df.loc[0, "Probability"]
    return top_category, top_confidence, prob_df


def main() -> None:
    st.title("🏦 Domain-Specific Document Classifier (Finance)")
    st.write(
        "Classify financial news, earnings releases, analyst notes, and market reports into **20 financial categories** using a fine-tuned Transformer."
    )

    try:
        tokenizer, model = load_classifier()
    except Exception as exc:
        st.error(
            f"Could not load fine-tuned model from `{MODEL_DIR}`. "
            f"Please run `python -m src.train_classifier` first.\n\nDetails: {exc}"
        )
        return

    with st.sidebar:
        st.header("Model Info")
        st.write(f"**Backbone:** `{model.config._name_or_path}`")
        st.write(f"**Categories:** `{model.config.num_labels}` Finance Topics")
        summary_sentences = st.slider("Summary Length (sentences)", min_value=1, max_value=5, value=3)

        st.divider()
        st.subheader("Try a Sample Snippet")
        sample_choice = st.selectbox(
            "Load example text:",
            [
                "Custom Input",
                "Q3 Earnings Beat: Apple reports quarterly revenue of $89.5 billion, beating Wall Street EPS estimates on strong services growth.",
                "Federal Reserve Chair Jerome Powell signals interest rates may remain higher for longer to bring inflation back to the 2% target.",
                "Microsoft announces a $68.7 billion all-cash acquisition deal following regulatory antitrust approval.",
            ],
        )

    tab_text, tab_file = st.tabs(["✍️ Enter Text", "📄 Upload Document (.txt, .pdf, .docx)"])

    raw_text = ""
    with tab_text:
        default_val = "" if sample_choice == "Custom Input" else sample_choice
        pasted_text = st.text_area(
            "Paste financial document or news snippet:",
            value=default_val,
            height=180,
            placeholder="Paste an earnings report excerpt, market update, or financial article here...",
        )
        if pasted_text.strip():
            raw_text = pasted_text

    with tab_file:
        uploaded_file = st.file_uploader(
            "Upload a financial document",
            type=["txt", "pdf", "docx"],
        )
        if uploaded_file is not None:
            extracted = extract_text_from_upload(uploaded_file)
            if extracted.strip():
                raw_text = extracted
                with st.expander("Preview Extracted Document Text"):
                    st.write(raw_text[:2000] + ("..." if len(raw_text) > 2000 else ""))
            else:
                st.warning("Could not extract readable text from the uploaded file.")

    classify_btn = st.button("🔍 Classify Document", type="primary")

    if classify_btn:
        if not raw_text.strip():
            st.warning("Please enter financial text or upload a supported document before classifying.")
            return

        with st.spinner("Running Transformer classification..."):
            category, confidence, prob_df = predict_document(raw_text, tokenizer, model)
            summary = extractive_summary(raw_text, num_sentences=summary_sentences)

        st.divider()
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Predicted Financial Category", category)
        with m2:
            st.metric("Confidence Score", f"{confidence * 100:.2f}%")

        st.subheader("📝 Document Summary")
        st.info(summary if summary else "Document is concise enough to read directly.")

        st.subheader("📊 Top 5 Category Probabilities")
        top5 = prob_df.head(5).copy()
        top5["Confidence (%)"] = (top5["Probability"] * 100).round(2)

        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.barh(top5["Category"][::-1], top5["Confidence (%)"][::-1])
            ax.set_xlabel("Confidence (%)")
            ax.set_xlim(0, 100)
            ax.set_title("Top 5 Predicted Categories")
            plt.tight_layout()
            st.pyplot(fig)

        with c2:
            st.dataframe(
                top5[["Category", "Confidence (%)"]],
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()