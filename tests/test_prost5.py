import os
import sys
import pytest
import torch
import json
from datetime import datetime

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.prost5 import ProstT5

def save_results(results, filename):
    """Save results to a JSON file."""
    # Create results directory if it doesn't exist
    results_dir = os.path.join(project_root, 'test_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    # Convert torch tensors to lists for JSON serialization
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                results[key] = value.tolist()
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {filepath}")
    return filepath

def test_prost5_initialization():
    """Test if the model initializes correctly."""
    model = ProstT5()
    assert model is not None
    assert model.device in ['cpu', 'cuda']

def test_prost5_embeddings():
    """Test embedding extraction functionality."""
    model = ProstT5()
    
    # Test sequences
    sequences = [
        "MLLAVLYCLAVFALSLPGK",  # A short protein sequence
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"  # A longer sequence
    ]
    
    # Test per-residue embeddings
    embeddings = model.get_embeddings(sequences, per_protein=False)
    assert embeddings.shape[0] == len(sequences)  # Batch size
    assert embeddings.shape[2] == 1024  # Embedding dimension
    
    # Test per-protein embeddings
    protein_embeddings = model.get_embeddings(sequences, per_protein=True)
    assert protein_embeddings.shape[0] == len(sequences)  # Batch size
    assert protein_embeddings.shape[1] == 1024  # Embedding dimension
    
    # Save results
    results = {
        'sequences': sequences,
        'per_residue_embeddings': embeddings,
        'per_protein_embeddings': protein_embeddings
    }
    save_results(results, 'embeddings_test')

def test_prost5_sequence_to_structure():
    """Test sequence to structure translation."""
    model = ProstT5()
    
    # Test sequences
    sequences = [
        "MLLAVLYCLAVFALSLPGK",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    ]
    
    # Translate to structure
    structures = model.sequence_to_structure(sequences)
    
    # Check results
    assert len(structures) == len(sequences)
    assert all(isinstance(s, str) for s in structures)
    assert all(len(s) > 0 for s in structures)
    
    # Save results
    results = {
        'input_sequences': sequences,
        'output_structures': structures
    }
    save_results(results, 'sequence_to_structure_test')

def test_prost5_structure_to_sequence():
    """Test structure to sequence translation."""
    model = ProstT5()
    
    # Test structure sequences (3Di format)
    structures = [
        "abcdefghijklmnopqrst",  # Example 3Di structure
        "uvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
    ]
    
    # Translate to sequences
    sequences = model.structure_to_sequence(structures)
    
    # Check results
    assert len(sequences) == len(structures)
    assert all(isinstance(s, str) for s in sequences)
    assert all(len(s) > 0 for s in sequences)
    
    # Save results
    results = {
        'input_structures': structures,
        'output_sequences': sequences
    }
    save_results(results, 'structure_to_sequence_test')

def test_prost5_roundtrip():
    """Test roundtrip translation (sequence -> structure -> sequence)."""
    model = ProstT5()
    
    # Original sequence
    original_sequence = "MLLAVLYCLAVFALSLPGK"
    
    # Sequence to structure
    structure = model.sequence_to_structure([original_sequence])[0]
    
    # Structure back to sequence
    back_translated = model.structure_to_sequence([structure])[0]
    
    # Check results
    assert isinstance(structure, str)
    assert isinstance(back_translated, str)
    assert len(structure) > 0
    assert len(back_translated) > 0
    
    # Save results
    results = {
        'original_sequence': original_sequence,
        'intermediate_structure': structure,
        'back_translated_sequence': back_translated
    }
    save_results(results, 'roundtrip_test')

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__]) 