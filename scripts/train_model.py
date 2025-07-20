import json
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments

# Load your dataset
with open("data/bangla_qa.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)
    data = json_data["data"]

# Initialize tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
model = BertForQuestionAnswering.from_pretrained("bert-base-multilingual-cased")

contexts = [item["context"] for item in data]
questions = [item["question"] for item in data]
answers = [item["answers"] for item in data]

# Tokenize with return_offsets_mapping=True and return sequence ids
encodings = tokenizer(
    questions,
    contexts,
    truncation=True,
    padding=True,
    max_length=384,
    return_offsets_mapping=True,
)

start_positions = []
end_positions = []

for i in range(len(data)):
    answer = answers[i]["text"][0]
    start_char = answers[i]["answer_start"][0]
    end_char = start_char + len(answer)

    offsets = encodings["offset_mapping"][i]        # offsets for the i-th example
    sequence_ids = encodings.sequence_ids(i)        # sequence ids for the i-th example

    start_token = None
    end_token = None

    # Only look at context tokens (sequence_id == 1)
    for idx, seq_id in enumerate(sequence_ids):
        if seq_id == 1:
            start, end = offsets[idx]
            if start <= start_char < end:
                start_token = idx
            if start < end_char <= end:
                end_token = idx

    # If answer span is not found in tokenized context, set default
    if start_token is None:
        start_token = 0
    if end_token is None:
        end_token = 0

    start_positions.append(start_token)
    end_positions.append(end_token)

# Remove offset_mapping before passing to model
encodings.pop("offset_mapping")

# Add start and end positions to the encodings
encodings["start_positions"] = start_positions
encodings["end_positions"] = end_positions

# Dataset class for Trainer
class QADataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])

dataset = QADataset(encodings)

# Training arguments
training_args = TrainingArguments(
    output_dir="model",
    num_train_epochs=3,               # You can increase this for better results
    per_device_train_batch_size=4,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    save_total_limit=1,
    save_safetensors=False,           # <-- Disable safetensors to avoid Windows file locking error
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
trainer.save_model("model")
tokenizer.save_pretrained("model")
