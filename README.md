# Bangla QA Chatbot - রসায়নের ধারণা (The Concepts of Chemistry)

## Overview
This is a Bengali question-answering chatbot trained specifically on the chemistry textbook chapter **"রসায়নের ধারণা"** (The Concepts of Chemistry) designed for 9th-10th grade students in the science stream. The model is capable of understanding and answering questions related to the concepts covered in this chapter.

## Training Data
The chatbot is trained on the complete text of the chapter which includes:

- Introduction to chemistry and its definition  
- Historical development of chemistry  
- Scope and fields of chemistry  
- Relationship between chemistry and other branches of science  
- Importance of studying chemistry  
- Research process in chemistry  
- Safety measures in chemistry laboratories  

## Capabilities
The chatbot can answer questions regarding:

- Basic concepts of chemistry  
- Historical figures in chemistry (e.g., Jabir ibn Hayyan, Lavoisier)  
- Chemistry's connections to biology, physics, and mathematics  
- Importance of chemistry in daily life  
- Laboratory safety procedures  
- Chemical symbols and their meanings  
- Research methodology in chemistry  

## Example Questions
- **রসায়ন কী?** (What is chemistry?)  
- **রসায়নের জনক কে?** (Who is the father of chemistry?)  
- **রসায়ন পরীক্ষাগার কী?** (What is the importance of studying chemistry?)  
- **সেফটি গগলস কেন ব্যবহার করা হয়?** (What safety measures should be taken in the laboratory?)  

## Limitations
- The chatbot only answers questions based on the content of the trained chapter.  
- It may not accurately handle questions outside the scope of this chapter.  
- Performance depends on how closely the question matches the trained material.  

## Technology Used
- **Python** — programming language for building the chatbot backend.  
- **Transformers (Hugging Face)** — to load and use pre-trained BERT models fine-tuned for question answering in Bengali.  
- **PyTorch** — deep learning framework used to run the model inference.  
- **Streamlit** — for creating an interactive web UI to chat with the model easily.  
- **JSON** — for storing the training dataset (chapter text and QA pairs).  
- **Google Colab / Local GPU** — used for model training and fine-tuning.  

## Live Demo
Try the chatbot live here:  
[https://f8cwc56ssgcfcrgappdbsew.streamlit.app/](https://f8cwc56ssgcfcrgappdbsew.streamlit.app/)

## Usage
Simply ask any question in Bengali related to the concepts covered in the chapter **"রসায়নের ধারণা"** and the chatbot will provide answers based on the textbook content.
