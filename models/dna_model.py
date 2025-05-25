import os
import requests
from typing import Dict, Any, Optional
import json

class DNALanguageModel:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the DNA model with NVIDIA API credentials for Evo2."""
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA API key is required")
        
        self.api_base_url = os.getenv("NVIDIA_API_BASE_URL", "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions")
        self.model_id = os.getenv("EVO2_MODEL_ID", "evo2-dna-model-id")  # Evo2 specific model ID
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def preprocess_sequence(self, sequence: str) -> str:
        """Preprocess DNA sequence for Evo2 model input."""
        # Evo2 specific preprocessing
        sequence = sequence.upper().strip()
        # Remove any non-DNA characters
        sequence = ''.join(c for c in sequence if c in 'ATCG')
        return sequence

    def predict(self, sequence: str) -> Dict[str, Any]:
        """Make predictions on a DNA sequence using Evo2."""
        sequence = self.preprocess_sequence(sequence)
        
        payload = {
            "inputs": [
                {
                    "sequence": sequence,
                    "parameters": {
                        "max_length": 512,
                        "temperature": 0.7,
                        "model": "evo2"  # Specify Evo2 model
                    }
                }
            ]
        }

        try:
            response = requests.post(
                f"{self.api_base_url}/{self.model_id}",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling NVIDIA Evo2 API: {str(e)}")

    def batch_predict(self, sequences: list[str]) -> list[Dict[str, Any]]:
        """Make batch predictions on multiple DNA sequences using Evo2."""
        processed_sequences = [self.preprocess_sequence(seq) for seq in sequences]
        
        payload = {
            "inputs": [
                {
                    "sequence": seq,
                    "parameters": {
                        "max_length": 512,
                        "temperature": 0.7,
                        "model": "evo2"  # Specify Evo2 model
                    }
                } for seq in processed_sequences
            ]
        }

        try:
            response = requests.post(
                f"{self.api_base_url}/{self.model_id}/batch",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling NVIDIA Evo2 API: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Evo2 model from NVIDIA's API."""
        try:
            response = requests.get(
                f"{self.api_base_url}/{self.model_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error getting Evo2 model info: {str(e)}")

    def fine_tune(self, training_data, validation_data=None):
        """Fine-tune the model on specific data."""
        # Implement fine-tuning logic here
        pass 