import streamlit as st
import json
from transformers import BertTokenizerFast, BertForQuestionAnswering, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np

@st.cache_resource
def load_model():
    repo_id = "robiulhasanjisan88/Bangla-QA-BERT"
    tokenizer = BertTokenizerFast.from_pretrained(repo_id)
    model = BertForQuestionAnswering.from_pretrained(repo_id)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_embedder():
    # Embedding extractor model (BERT base for embeddings)
    embedder_model_name = "sagorsarker/bangla-bert-base"
    tokenizer = BertTokenizerFast.from_pretrained(embedder_model_name)
    model = AutoModel.from_pretrained(embedder_model_name)
    model.eval()
    return tokenizer, model

@st.cache_data
def load_qa_data():
    with open("D:/Aimodel/data/bangla_qa.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data["data"]

def normalize_text(text):
    text = text.lower()
    # বানান ও শব্দ পরিবর্তন
    text = re.sub(r"কে|কার|কার দ্বারা", "কে", text)
    text = re.sub(r"দিয়েছিল|দেয়েছিল|দেয়া হয়েছিল", "দেয়েছিল", text)
    text = re.sub(r"প্রথমে|শুরুতে", "", text)
    text = re.sub(r"\s+", " ", text).strip()  # অতিরিক্ত স্পেস মুছে ফেলুন
    return text

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    # CLS টোকেনের embedding নেওয়া
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding

def find_best_match_question(user_question, qa_data, tokenizer_embedder, model_embedder):
    user_question_norm = normalize_text(user_question)
    user_emb = get_embedding(user_question_norm, tokenizer_embedder, model_embedder)

    best_qa = None
    best_score = -1

    for qa in qa_data:
        qa_question_norm = normalize_text(qa["question"])
        qa_emb = get_embedding(qa_question_norm, tokenizer_embedder, model_embedder)

        score = cosine_similarity(user_emb, qa_emb)[0][0]
        if score > best_score:
            best_score = score
            best_qa = qa

    # Similarity threshold দিয়ে নিশ্চিত হোন
    if best_score > 0.80:
        return best_qa
    else:
        return None

def chatbot_answer(user_question, qa_data, tokenizer_embedder, model_embedder):
    qa_pair = find_best_match_question(user_question, qa_data, tokenizer_embedder, model_embedder)
    if qa_pair:
        return qa_pair["answers"]["text"][0]
    else:
        return "দুঃখিত, আপনার প্রশ্নের জন্য সঠিক উত্তর পাওয়া যায়নি।"

# Load models & data
tokenizer_qa, model_qa = load_model()
tokenizer_embedder, model_embedder = load_embedder()
qa_data = load_qa_data()

# Streamlit UI code

st.set_page_config(page_title="Bangla QA Chatbot", page_icon="", layout="centered")

st.markdown("""<style>/* ...your CSS styles here... */</style>""", unsafe_allow_html=True)

st.title("Bangla QA Chatbot")
st.write("বাংলা ভাষায় আপনার প্রশ্ন করুন এবং সঠিক উত্তর পান।")

if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([4,1])
with col1:
    user_question = st.text_input("আপনার প্রশ্ন লিখুন", key="input_question", placeholder="এখানে আপনার প্রশ্ন টাইপ করুন...")
with col2:
    if st.button("উত্তর পান"):
        if user_question.strip():
            with st.spinner("আপনার প্রশ্নের উত্তর খুঁজছি..."):
                answer = chatbot_answer(user_question, qa_data, tokenizer_embedder, model_embedder)
                st.session_state.history.append({"question": user_question, "answer": answer})
        else:
            st.warning("অনুগ্রহ করে প্রশ্ন লিখুন।")

if st.button("সাফ করুন"):
    st.session_state.history = []

if st.session_state.history:
    st.markdown("### পূর্ববর্তী প্রশ্ন ও উত্তর")
    for qa in reversed(st.session_state.history[-10:]):
        st.markdown(f"**প্রশ্ন:** {qa['question']}")
        st.info(f"উত্তর: {qa['answer']}")

with st.expander("মডেলের বিস্তারিত তথ্য"):
    st.write("""
    - BERT Question Answering Model
    - Fine-tuned on Bangla QA dataset
    - Powered by Huggingface Transformers
    """)
