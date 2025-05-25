# LLM Research Tool Usage Guide

This guide provides examples of how to use the LLM Research Tool API for DNA and protein sequence analysis.

## Setup

1. Set up your environment variables:
```bash
# NVIDIA API (Evo2)
NVIDIA_API_KEY=your-nvidia-api-key
NVIDIA_API_BASE_URL=https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions
EVO2_MODEL_ID=your-evo2-model-id

# Hugging Face API (ESM-1b)
HUGGINGFACE_API_KEY=your-huggingface-api-key
```

2. Start the API server:
```bash
python -m api.main
```

## API Endpoints

### DNA Sequence Analysis (Evo2)

#### Single Sequence Prediction
```python
import requests

# Predict single DNA sequence
response = requests.post(
    "http://localhost:8000/dna/predict",
    json={"sequence": "ATCGATCGATCG"}
)
result = response.json()
print(result)
```

Example response:
```json
{
    "sequence": "ATCGATCGATCG",
    "predictions": {
        "embeddings": [...],
        "structure": "..."
    }
}
```

#### Batch Sequence Prediction
```python
# Predict multiple DNA sequences
response = requests.post(
    "http://localhost:8000/dna/batch-predict",
    json={
        "sequences": [
            "ATCGATCGATCG",
            "GCTAGCTAGCTA"
        ]
    }
)
results = response.json()
print(results)
```

### Protein Sequence Analysis (ESM-1b)

#### Single Sequence Prediction
```python
# Predict single protein sequence
response = requests.post(
    "http://localhost:8000/protein/predict",
    json={"sequence": "MLLAVLYCLLW"}
)
result = response.json()
print(result)
```

Example response:
```json
{
    "sequence": "MLLAVLYCLLW",
    "predictions": {
        "embeddings": [...],
        "structure": "HHH",
        "sequence": "MLLAVLYCLLW"
    }
}
```

#### Get Protein Embeddings
```python
# Get protein sequence embeddings
response = requests.get(
    "http://localhost:8000/protein/embeddings",
    params={"sequence": "MLLAVLYCLLW"}
)
result = response.json()
print(result)
```

Example response:
```json
{
    "sequence": "MLLAVLYCLLW",
    "embeddings": [...]
}
```

#### Batch Protein Prediction
```python
# Predict multiple protein sequences
response = requests.post(
    "http://localhost:8000/protein/batch-predict",
    json={
        "sequences": [
            "MLLAVLYCLLW",
            "ACDEFGHIKLM"
        ]
    }
)
results = response.json()
print(results)
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- 200: Successful request
- 400: Invalid request (e.g., malformed sequence)
- 500: Server error (e.g., API key issues, model errors)

Example error response:
```json
{
    "detail": "Error calling ESM-1b API: Invalid sequence format"
}
```

## Best Practices

1. **Sequence Validation**
   - DNA sequences should only contain A, T, C, G
   - Protein sequences should only contain standard amino acid codes

2. **Batch Processing**
   - Use batch endpoints for multiple sequences
   - Keep batch sizes reasonable (recommended: 10-50 sequences)

3. **Error Handling**
   - Always check response status codes
   - Handle API errors gracefully
   - Implement retry logic for transient failures

4. **Performance**
   - Cache frequently used sequences
   - Use batch endpoints for multiple predictions
   - Consider implementing rate limiting for high-volume usage

## Rate Limits

- NVIDIA API: Check your API plan for specific limits
- Hugging Face API: Check your API plan for specific limits

## Support

For issues or questions:
1. Check the API documentation
2. Review error messages
3. Contact support with detailed error information 