import pytest
from unittest.mock import patch, MagicMock
from models.dna_model import DNALanguageModel

@pytest.fixture
def mock_api_key():
    return "test-api-key"

@pytest.fixture
def dna_model(mock_api_key):
    with patch.dict('os.environ', {'NVIDIA_API_KEY': mock_api_key}):
        return DNALanguageModel()

def test_dna_model_initialization(mock_api_key):
    with patch.dict('os.environ', {'NVIDIA_API_KEY': mock_api_key}):
        model = DNALanguageModel()
        assert model is not None
        assert model.api_key == mock_api_key
        assert model.headers["Authorization"] == f"Bearer {mock_api_key}"

def test_preprocess_sequence(dna_model):
    sequence = "atcg"
    processed = dna_model.preprocess_sequence(sequence)
    assert isinstance(processed, str)
    assert processed == "ATCG"

@patch('requests.post')
def test_predict(mock_post, dna_model):
    # Mock API response
    mock_response = MagicMock()
    mock_response.json.return_value = {"predictions": [{"label": "test", "score": 0.9}]}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    sequence = "ATCG"
    result = dna_model.predict(sequence)
    
    assert result == {"predictions": [{"label": "test", "score": 0.9}]}
    mock_post.assert_called_once()

@patch('requests.post')
def test_batch_predict(mock_post, dna_model):
    # Mock API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "predictions": [
            {"label": "test1", "score": 0.9},
            {"label": "test2", "score": 0.8}
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    sequences = ["ATCG", "GCTA"]
    result = dna_model.batch_predict(sequences)
    
    assert result == {
        "predictions": [
            {"label": "test1", "score": 0.9},
            {"label": "test2", "score": 0.8}
        ]
    }
    mock_post.assert_called_once()

@patch('requests.get')
def test_get_model_info(mock_get, dna_model):
    # Mock API response
    mock_response = MagicMock()
    mock_response.json.return_value = {"model_info": "test_info"}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = dna_model.get_model_info()
    
    assert result == {"model_info": "test_info"}
    mock_get.assert_called_once() 