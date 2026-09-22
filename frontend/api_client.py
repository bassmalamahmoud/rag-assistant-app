import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Read API base URL from environment, never hardcoded
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

def check_backend_health() -> bool:
    """Checks if the FastAPI backend is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200
    except Exception:
        return False

def ask_backend(question: str) -> dict:
    """
    Sends user query to the FastAPI backend /query endpoint.
    Returns: {"answer": str, "sources": list[str]}
    Raises: Exception with readable message on failure
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    url = f"{API_BASE_URL}/query"
    payload = {"question": question.strip()}

    try:
        response = requests.post(url, json=payload, timeout=45)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Could not connect to backend at {API_BASE_URL}. Ensure the FastAPI server is running."
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("The backend query request timed out. Please try again.")
    except requests.exceptions.HTTPError as err:
        detail = response.text
        raise RuntimeError(f"Backend returned HTTP {response.status_code}: {detail}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error communicating with backend: {str(e)}")
