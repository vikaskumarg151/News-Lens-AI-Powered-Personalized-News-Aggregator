
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pretrained model & tokenizer
MODEL_NAME = "mrm8488/bert-tiny-finetuned-fake-news"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def classify_news_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    labels = ["real", "fake"]  # model-specific labels
    return labels[pred], probs[0][pred].item()

# Test
headline = "White House releases Gaza peace plan as Trump meets Netanyahu"
label, confidence = classify_news_bert(headline)
print(f"Headline: {headline}")
print(f"Prediction: {label} (confidence: {confidence:.2f})")
