import pytest
from unittest.mock import patch, MagicMock
from models.protein_model import ProteinLanguageModel

@pytest.fixture
def mock_api_key():
    return "test-huggingface-key"

@pytest.fixture
def protein_model(mock_api_key):
    with patch.dict('os.environ', {'HUGGINGFACE_API_KEY': mock_api_key}):
        return ProteinLanguageModel()

def test_protein_model_initialization(mock_api_key):
    with patch.dict('os.environ', {'HUGGINGFACE_API_KEY': mock_api_key}):
        model = ProteinLanguageModel()
        assert model is not None
        assert model.api_key == mock_api_key
        assert model.model_id == "facebook/esm-1b"

def test_preprocess_sequence(protein_model):
    # Test valid sequence
    sequence = "MLLAVLYCLLW"
    processed = protein_model.preprocess_sequence(sequence)
    assert isinstance(processed, str)
    assert processed == sequence

    # Test sequence with invalid characters
    sequence = "MLLAVLYCLLW123"
    processed = protein_model.preprocess_sequence(sequence)
    assert processed == "MLLAVLYCLLW"

    # Test empty sequence
    sequence = ""
    processed = protein_model.preprocess_sequence(sequence)
    assert processed == ""

@patch('huggingface_hub.InferenceClient.feature_extraction')
@patch('huggingface_hub.InferenceClient.text_generation')
def test_predict(mock_text_gen, mock_feature_extraction, protein_model):
    # Mock API responses
    mock_feature_extraction.return_value = [0.1, 0.2, 0.3]
    mock_text_gen.return_value = "HHH"

    sequence = "MLLAVLYCLLW"
    result = protein_model.predict(sequence)
    
    assert result["sequence"] == sequence
    assert result["embeddings"] == [0.1, 0.2, 0.3]
    assert result["structure"] == "HHH"
    
    mock_feature_extraction.assert_called_once()
    mock_text_gen.assert_called_once()

def test_batch_predict(protein_model):
    sequences = ["MLLAVLYCLLW", "ACDEFGHIKLM"]
    
    with patch.object(protein_model, 'predict') as mock_predict:
        mock_predict.side_effect = [
            {"embeddings": [0.1], "structure": "HHH", "sequence": "MLLAVLYCLLW"},
            {"embeddings": [0.2], "structure": "EEE", "sequence": "ACDEFGHIKLM"}
        ]
        
        results = protein_model.batch_predict(sequences)
        
        assert len(results) == 2
        assert results[0]["sequence"] == "MLLAVLYCLLW"
        assert results[1]["sequence"] == "ACDEFGHIKLM"
        assert mock_predict.call_count == 2

@patch('huggingface_hub.InferenceClient.model_info')
def test_get_model_info(mock_model_info, protein_model):
    # Mock API response
    mock_model_info.return_value = {
        "description": "ESM-1b model",
        "tags": ["protein", "biology"],
        "pipeline_tag": "feature-extraction"
    }
    
    result = protein_model.get_model_info()
    
    assert result["model_name"] == "ESM-1b"
    assert result["model_id"] == "facebook/esm-1b"
    assert result["description"] == "ESM-1b model"
    assert "protein" in result["tags"]
    mock_model_info.assert_called_once()

@patch('huggingface_hub.InferenceClient.feature_extraction')
def test_get_embeddings(mock_feature_extraction, protein_model):
    # Mock API response
    mock_feature_extraction.return_value = [0.1, 0.2, 0.3]
    
    sequence = "MLLAVLYCLLW"
    embeddings = protein_model.get_embeddings(sequence)
    
    assert embeddings is not None
    assert len(embeddings) == 3
    mock_feature_extraction.assert_called_once()

def test_error_handling(protein_model):
    # Test missing API key
    with pytest.raises(ValueError):
        ProteinLanguageModel(api_key=None)
    
    # Test API error
    with patch.object(protein_model.client, 'feature_extraction') as mock_feature_extraction:
        mock_feature_extraction.side_effect = Exception("API Error")
        with pytest.raises(Exception) as exc_info:
            protein_model.predict("MLLAVLYCLLW")
        assert "Error calling ESM-1b API" in str(exc_info.value) 