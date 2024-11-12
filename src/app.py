from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pickle
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

# Load tokenizer and model at startup
try:
    with open("src/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("Tokenizer file not found at src/tokenizer.pkl")

# Initialize model and force CPU usage
device = torch.device("cpu")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

try:
    # Load the model weights with explicit CPU mapping
    state_dict = torch.load("src/model.pt", map_location=device)
    model.load_state_dict(state_dict)
except FileNotFoundError:
    raise RuntimeError("Model file not found at src/model.pt")
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

model.to(device)
model.eval()

# Define input schema
class TextInput(BaseModel):
    sentence: str

@app.post("/predict")
async def predict_sarcasm(input: TextInput):
    try:
        inputs = tokenizer(
            input.sentence,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            scores = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1).item()

        label_map = {0: "Not Sarcastic", 1: "Sarcastic"}
        sarcasm_score = scores[0, 1].item()

        # Return response as JSON
        return {
            "prediction": label_map[prediction],
            "sarcasm_score": sarcasm_score,
            "input_text": input.sentence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}