from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

MODEL_PATH = "./models/sentiment_model"

# Carregar modelo
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

app = FastAPI(title="Sentiment Analysis API")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input_data: TextInput):
    try:
        inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            score, pred = torch.max(probs, dim=1)
        label = model.config.id2label[pred.item()]
        return {"sentiment": label, "score": round(score.item(), 3)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
