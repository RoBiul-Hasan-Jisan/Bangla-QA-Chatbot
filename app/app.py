import streamlit as st
import json
from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch

@st.cache_resource
def load_model():
    # Load directly from Hugging Face Hub
    repo_id = "robiulhasanjisan88/Bangla-QA-BERT"
    tokenizer = BertTokenizerFast.from_pretrained(repo_id)
    model = BertForQuestionAnswering.from_pretrained(repo_id)
    model.eval()
    return tokenizer, model

@st.cache_data
def load_qa_data():
    with open("D:/Aimodel/data/bangla_qa.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data["data"]

def find_best_match_question(user_question, qa_data):
    user_question = user_question.strip()
    # Exact match first
    for qa in qa_data:
        if qa["question"].strip() == user_question:
            return qa
    # Substring match fallback
    for qa in qa_data:
        if user_question in qa["question"]:
            return qa
    return None

def chatbot_answer(user_question, qa_data):
    qa_pair = find_best_match_question(user_question, qa_data)
    if qa_pair:
        return qa_pair["answers"]["text"][0]
    else:
        return "দুঃখিত, আপনার প্রশ্নের জন্য সঠিক উত্তর পাওয়া যায়নি।"

# Load model & data
tokenizer, model = load_model()
qa_data = load_qa_data()

st.set_page_config(page_title="Bangla QA Chatbot", page_icon="", layout="centered")

st.markdown("""
<style>
/* Customize input box */
[data-testid="stTextInput"] > div > input {
    font-size: 20px;
    padding: 10px;
}
/* Customize button */
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    padding: 10px 25px;
    border-radius: 8px;
}
div.stButton > button:hover {
    background-color: #45a049;
}
/* Customize success message */
div.stAlert.success {
    font-size: 18px;
    background-color: #e6f4ea;
    border-left: 6px solid #4CAF50;
}
/* Customize warning message */
div.stAlert.warning {
    font-size: 18px;
    background-color: #fff4e5;
    border-left: 6px solid #ffb347;
}
</style>
""", unsafe_allow_html=True)

st.title("Bangla QA Chatbot")
st.write("বাংলা ভাষায় আপনার প্রশ্ন করুন এবং সঠিক উত্তর পান।")

# Session state to keep Q&A history
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([4,1])

with col1:
    user_question = st.text_input("আপনার প্রশ্ন লিখুন", key="input_question", placeholder="এখানে আপনার প্রশ্ন টাইপ করুন...")
with col2:
    if st.button("উত্তর পান"):
        if user_question.strip():
            with st.spinner("আপনার প্রশ্নের উত্তর খুঁজছি..."):
                answer = chatbot_answer(user_question, qa_data)
                st.session_state.history.append({"question": user_question, "answer": answer})
        else:
            st.warning("অনুগ্রহ করে প্রশ্ন লিখুন।")

if st.button("সাফ করুন"):
    st.session_state.history = []

# Show Q&A history
if st.session_state.history:
    st.markdown("### পূর্ববর্তী প্রশ্ন ও উত্তর")
    for qa in reversed(st.session_state.history[-10:]):
        st.markdown(f"**প্রশ্ন:** {qa['question']}")
        st.info(f"উত্তর: {qa['answer']}")

# Optionally show model info inside an expander
with st.expander("মডেলের বিস্তারিত তথ্য"):
    st.write("""
    - BERT Question Answering Model
    - Fine-tuned on Bangla QA dataset
    - Powered by Huggingface Transformers
    """)










   
