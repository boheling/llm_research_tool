import os
import sys
import pytest
import torch
import json
from datetime import datetime
from Bio import SeqIO

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.prost5 import ProstT5

def read_fasta_sequences(fasta_path):
    """Read sequences from a FASTA file."""
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences

def save_results(results, filename):
    """Save results to a JSON file."""
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), 'test_results')
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
    print("Model initialized successfully: ", model.device)

def test_prost5_embeddings():
    """Test embedding extraction functionality."""
    model = ProstT5()
    
    # Read sequences from FASTA file
    fasta_path = os.path.join(os.path.dirname(__file__), 'test_data', 'input_aa.fasta')
    sequences = read_fasta_sequences(fasta_path)
    print(f"Read {len(sequences)} sequences from {fasta_path}")
    
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
    
    # Read sequences from FASTA file
    fasta_path = os.path.join(os.path.dirname(__file__), 'test_data', 'input_aa.fasta')
    sequences = read_fasta_sequences(fasta_path)
    print(f"Read {len(sequences)} sequences from {fasta_path}")
    
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
    
    # Read structures directly from FASTA file
    fasta_path = os.path.join(os.path.dirname(__file__), 'test_data', 'input_structure.fasta')
    structures = read_fasta_sequences(fasta_path)
    print(f"Read {len(structures)} structures from {fasta_path}")
    
    # Translate structures to sequences
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

def calculate_sequence_accuracy(original, predicted):
    """Calculate sequence accuracy metrics.
    
    Args:
        original (str): Original amino acid sequence
        predicted (str): Predicted amino acid sequence
        
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    if len(original) != len(predicted):
        return {
            'exact_match': 0.0,
            'per_residue_accuracy': 0.0,
            'length_mismatch': True
        }
    
    exact_match = original == predicted
    correct_residues = sum(1 for a, b in zip(original, predicted) if a == b)
    per_residue_accuracy = correct_residues / len(original)
    
    return {
        'exact_match': float(exact_match),
        'per_residue_accuracy': per_residue_accuracy,
        'length_mismatch': False
    }

def test_prost5_roundtrip():
    """Test roundtrip translation (sequence -> structure -> sequence)."""
    model = ProstT5()
    
    # Read sequences from FASTA file
    fasta_path = os.path.join(os.path.dirname(__file__), 'test_data', 'input_aa.fasta')
    sequences = read_fasta_sequences(fasta_path)
    print(f"Read {len(sequences)} sequences from {fasta_path}")
    
    # Process each sequence
    results = []
    total_exact_matches = 0
    total_per_residue_accuracy = 0
    
    for seq in sequences:
        # Sequence to structure
        structure = model.sequence_to_structure([seq])[0]
        
        # Structure back to sequence
        back_translated = model.structure_to_sequence([structure])[0]
        
        # Calculate accuracy
        accuracy_metrics = calculate_sequence_accuracy(seq, back_translated)
        total_exact_matches += accuracy_metrics['exact_match']
        total_per_residue_accuracy += accuracy_metrics['per_residue_accuracy']
        
        # Check results
        assert isinstance(structure, str)
        assert isinstance(back_translated, str)
        assert len(structure) > 0
        assert len(back_translated) > 0
        
        results.append({
            'original_sequence': seq,
            'intermediate_structure': structure,
            'back_translated_sequence': back_translated,
            'accuracy_metrics': accuracy_metrics
        })
    
    # Calculate overall metrics
    num_sequences = len(sequences)
    overall_metrics = {
        'exact_match_rate': total_exact_matches / num_sequences,
        'average_per_residue_accuracy': total_per_residue_accuracy / num_sequences
    }
    
    # Save results
    save_results({
        'roundtrip_results': results,
        'overall_metrics': overall_metrics
    }, 'roundtrip_test')

def test_prost5_reverse_roundtrip():
    """Test reverse roundtrip: structure -> sequence -> structure."""
    model = ProstT5()

    # Prepare structure examples
    fasta_path = os.path.join(os.path.dirname(__file__), 'test_data', 'input_structure.fasta')
    structures = read_fasta_sequences(fasta_path)
    # Add canonical alpha helix and beta sheet
    structures += [
        'h' * 100,  # alpha helix
        'e' * 100   # beta sheet
    ]
    print(f"Testing {len(structures)} structures (including canonical examples)")

    results = []
    total_exact_matches = 0
    total_per_residue_accuracy = 0

    for struct in structures:
        # Structure to sequence
        aa_seq = model.structure_to_sequence([struct])[0]
        # Sequence back to structure
        struct_back = model.sequence_to_structure([aa_seq])[0]
        # Calculate accuracy
        accuracy_metrics = calculate_sequence_accuracy(struct, struct_back)
        total_exact_matches += accuracy_metrics['exact_match']
        total_per_residue_accuracy += accuracy_metrics['per_residue_accuracy']
        results.append({
            'original_structure': struct,
            'intermediate_sequence': aa_seq,
            'back_translated_structure': struct_back,
            'accuracy_metrics': accuracy_metrics
        })

    num_structures = len(structures)
    overall_metrics = {
        'exact_match_rate': total_exact_matches / num_structures,
        'average_per_residue_accuracy': total_per_residue_accuracy / num_structures
    }
    save_results({
        'reverse_roundtrip_results': results,
        'overall_metrics': overall_metrics
    }, 'reverse_roundtrip_test')

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__]) 