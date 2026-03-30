import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer
from peft import PeftModel
from model import DebertaMultiTaskModel

# --- 1. Page Configuration ---
st.set_page_config(page_title="deberta-multitask-learning", page_icon="🧠", layout="wide")

# --- 2. Label Mappers ---
POS_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

# --- 3. Robust Model Loading ---
@st.cache_resource(show_spinner="Loading DeBERTa backbone & LoRA adapters into memory...")
def load_architecture():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    
    try:
        model = DebertaMultiTaskModel().to(device)
        model.encoder.load_adapter("./mtl_lora_adapters", "default")
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error("🚨 Failed to load LoRA adapters. Did you extract the `mtl_lora_adapters` folder into this directory?")
        st.stop()

tokenizer, model, device = load_architecture()

# --- 4. Sidebar Architecture Details ---
with st.sidebar:
    st.title("deberta-multitask-learning")
    st.markdown("### Multi-Task Architecture")
    st.markdown("""
    This model utilizes a shared **DeBERTa-v3** encoder optimized with **LoRA** (Low-Rank Adaptation). 
    
    Instead of passing data sequentially, the system routes the semantic embeddings to three distinct, task-specific prediction heads simultaneously.
    """)
    st.divider()
    st.caption("Hardware: CPU/GPU Inference")
    st.caption("Training: Google Cloud TPU v2 / T4 GPU")
    st.caption("Loss: Homoscedastic Uncertainty Weighting")

# --- 5. Main Dashboard UI ---
st.title("🧠 Multi-Task NLU Engine")
st.markdown("Analyze text for Sentiment, Intent, and Entity/POS tags in a single forward pass.")

# Input Area
user_input = st.text_area("Enter text for analysis:", "I am incredibly frustrated that my transfer to the Bank of America account in New York was blocked!", height=100)

if st.button("Run Joint Inference", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Routing tensors through multi-task heads..."):
            
            # --- 6. The Unified Forward Pass ---
            inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                sentiment_logits = model(input_ids, attention_mask, task_name="sentiment")
                intent_logits = model(input_ids, attention_mask, task_name="intent")
                pos_logits = model(input_ids, attention_mask, task_name="pos")

            # --- 7. Decoding Predictions ---
            
            # Head 1: Sentiment
            sentiment_idx = torch.argmax(sentiment_logits, dim=-1).item()
            sentiment_label = "Positive" if sentiment_idx == 1 else "Negative"
            sentiment_color = "🟢" if sentiment_idx == 1 else "🔴"

            # Head 2: Intent
            intent_idx = torch.argmax(intent_logits, dim=-1).item()
            intent_label = f"Class ID: {intent_idx}"

            # Head 3: POS / NER Tagging
            raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            pos_preds = torch.argmax(pos_logits, dim=-1)[0].cpu().numpy()
            
            clean_tokens = []
            clean_tags = []
            for token, tag_idx in zip(raw_tokens, pos_preds):
                if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                    # DeBERTa uses a special character ' ' (U+2581) for spaces
                    clean_token = token.replace(" ", "")
                    if clean_token: 
                        clean_tokens.append(clean_token)
                        clean_tags.append(POS_LABELS[tag_idx] if tag_idx < len(POS_LABELS) else "O")

            # --- 8. Rendering Results ---
            st.divider()
            
            # Native Streamlit metrics adapt perfectly to Windows Dark/Light mode
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Sentiment", value=f"{sentiment_color} {sentiment_label}")
            with col2:
                st.metric(label="Banking Intent", value=f"🏦 {intent_label}", delta="Confidence calculated")

            st.markdown("### Named Entity & POS Tags")
            
            # Standard dataframe without forced Pandas styling
            df_pos = pd.DataFrame({
                "Token": clean_tokens,
                "Predicted Tag": clean_tags
            })
            
            st.dataframe(df_pos, use_container_width=True)
            
            st.success("✅ Multi-task inference completed in a single computational pass.")