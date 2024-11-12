# beforreal backend 🧐

![FastAPI](https://img.shields.io/badge/FastAPI-0.95.2-green)
![Docker](https://img.shields.io/badge/Docker-Supported-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

A FastAPI-based sarcasm detection API using a fine-tuned BERT model to predict sarcasm in text.

## Setup

### 1. Download Model and Tokenizer

Get `model.pt` and `tokenizer.pkl` from the [Releases](https://github.com/yourusername/sarcasm-detection-api/releases) section, and place them in the `src/` directory.

### 2. Run Locally

```bash
pip install -r requirements.txt
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

### 3. Run with Docker

```bash
docker build -t sarcasm-detection-api .
docker run -p 8000:8000 sarcasm-detection-api
```

## Usage

**Endpoint:** `POST /predict`

**Payload:**
```json
{
  "sentence": "I just love waiting in traffic all day!"
}
```

**Response:**
```json
{
  "prediction": "Sarcastic",
  "sarcasm_score": 0.87,
  "sarcasm": true
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"sentence": "Your input here"}'
```

## Testing

```bash
pytest test_api.py
```

## License

[MIT License](./LICENSE)
