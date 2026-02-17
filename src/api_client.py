import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Fetch API URL from environment variables
API_URL = os.getenv("API_URL", "http://localhost:5000")

def check_api_health():
    """Verify if the Flask API is reachable."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_prediction(features):
    """
    Send feature data to the Flask API and return the prediction.
    'features' should be a dictionary.
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=features,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}