import os
from transformers import BertForQuestionAnswering, BertTokenizerFast
from huggingface_hub import HfApi

model_dir = "model"  # local path to your model folder
repo_name = "Bangla-QA-BERT"  # name for your Hugging Face repo

token = os.environ.get("HUGGINGFACE_TOKEN")
if not token:
    raise ValueError("HUGGINGFACE_TOKEN environment variable not set!")

model = BertForQuestionAnswering.from_pretrained(model_dir)
tokenizer = BertTokenizerFast.from_pretrained(model_dir)

api = HfApi()
api.create_repo(repo_id=repo_name, token=token, exist_ok=True)

model.push_to_hub(repo_name, use_auth_token=token)
tokenizer.push_to_hub(repo_name, use_auth_token=token)

print(f"Model and tokenizer uploaded to https://huggingface.co/{api.whoami(token)['name']}/{repo_name}")
