import pytest
import httpx

BASE_URL = "http://localhost:8000"

test_cases = [
    ("I just love waiting in traffic all day!", True),  
    ("I'm really happy to see you!", False),            
    ("Oh great, another Monday.", True),                
    ("The sun is shining today.", False)                
]

@pytest.mark.parametrize("sentence, expected_sarcasm", test_cases)
def test_predict_sarcasm(sentence, expected_sarcasm):
    payload = {"sentence": sentence}
    response = httpx.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200, f"Failed for sentence: {sentence}"
    data = response.json()
    assert "prediction" in data, "Missing 'prediction' in response"
    assert "sarcasm_score" in data, "Missing 'sarcasm_score' in response"
    assert "sarcasm" in data, "Missing 'sarcasm' in response"
    assert data["sarcasm"] == expected_sarcasm, f"Expected sarcasm={expected_sarcasm} for sentence: {sentence}"
    assert 0.0 <= data["sarcasm_score"] <= 1.0, "Sarcasm score out of bounds"


def test_invalid_input():
    response = httpx.post(f"{BASE_URL}/predict", json={"sentence": ""})
    assert response.status_code == 422, "Expected a 422 error for empty input"
    response = httpx.post(f"{BASE_URL}/predict", json={})
    assert response.status_code == 422, "Expected a 422 error for missing 'sentence'"
