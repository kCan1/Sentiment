from fastapi import FastAPI, HTTPException
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from google.colab import userdata
userdata.get('sentiment')

app = FastAPI()

# Load pre-trained model and tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

@app.post("/analyze_sentiment/")
async def analyze_sentiment(text: str):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment = torch.argmax(probabilities, dim=-1).item()
        confidence = torch.max(probabilities, dim=-1).values.item()

        if sentiment == 1:
            sentiment_label = "Positive"
        else:
            sentiment_label = "Negative"

        return {"sentiment": sentiment_label, "confidence": confidence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
