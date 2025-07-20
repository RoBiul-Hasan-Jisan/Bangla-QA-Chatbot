import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering

# Load the model and tokenizer
model_dir = "../model"

tokenizer = BertTokenizerFast.from_pretrained(model_dir)
model = BertForQuestionAnswering.from_pretrained(model_dir)
model.eval()  # Ensure model is in evaluation mode

def answer_question(question, context):
    # Tokenize the question and context
    inputs = tokenizer.encode_plus(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Get start and end logits
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Get the most probable start and end positions
    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits) + 1

    # Convert token ids to string
    input_ids = inputs["input_ids"][0]
    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer.strip()

if __name__ == "__main__":
    context = "রসায়নের জনক হিসেবে অ্যান্টনি ল্যাভয়সিয়ে পরিচিত।"
    question = "রসায়নের জনক কে?"
    print("Question:", question)
    print("Answer:", answer_question(question, context))
