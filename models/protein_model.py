import os
from typing import Dict, Any, Optional
from huggingface_hub import InferenceClient
import torch

class ProteinLanguageModel:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the protein model with Hugging Face API credentials for ESM-1b."""
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("Hugging Face API key is required")
        
        self.model_id = "facebook/esm-1b"
        self.client = InferenceClient(
            model=self.model_id,
            token=self.api_key
        )

    def preprocess_sequence(self, sequence: str) -> str:
        """Preprocess protein sequence for ESM-1b model input."""
        # ESM-1b specific preprocessing
        sequence = sequence.strip()
        # Remove any non-protein characters
        sequence = ''.join(c for c in sequence if c in 'ACDEFGHIKLMNPQRSTVWY')
        return sequence

    def predict(self, sequence: str) -> Dict[str, Any]:
        """Make predictions on a protein sequence using ESM-1b."""
        sequence = self.preprocess_sequence(sequence)
        
        try:
            # Get embeddings from ESM-1b
            embeddings = self.client.feature_extraction(
                sequence,
                parameters={
                    "max_length": 1024,
                    "truncation": True
                }
            )
            
            # Get secondary structure prediction
            structure = self.client.text_generation(
                sequence,
                parameters={
                    "max_length": 1024,
                    "temperature": 0.7
                }
            )
            
            return {
                "embeddings": embeddings,
                "structure": structure,
                "sequence": sequence
            }
        except Exception as e:
            raise Exception(f"Error calling ESM-1b API: {str(e)}")

    def batch_predict(self, sequences: list[str]) -> list[Dict[str, Any]]:
        """Make batch predictions on multiple protein sequences using ESM-1b."""
        processed_sequences = [self.preprocess_sequence(seq) for seq in sequences]
        results = []
        
        for sequence in processed_sequences:
            try:
                result = self.predict(sequence)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "sequence": sequence})
        
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the ESM-1b model."""
        try:
            model_info = self.client.model_info()
            return {
                "model_name": "ESM-1b",
                "model_id": self.model_id,
                "description": model_info.get("description", ""),
                "tags": model_info.get("tags", []),
                "pipeline_tag": model_info.get("pipeline_tag", "")
            }
        except Exception as e:
            raise Exception(f"Error getting ESM-1b model info: {str(e)}")

    def get_embeddings(self, sequence: str) -> torch.Tensor:
        """Get protein sequence embeddings from ESM-1b."""
        sequence = self.preprocess_sequence(sequence)
        
        try:
            embeddings = self.client.feature_extraction(
                sequence,
                parameters={
                    "max_length": 1024,
                    "truncation": True
                }
            )
            return torch.tensor(embeddings)
        except Exception as e:
            raise Exception(f"Error getting ESM-1b embeddings: {str(e)}") 