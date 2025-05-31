import torch
from transformers import T5Tokenizer, T5EncoderModel, AutoModelForSeq2SeqLM
import re
from typing import List, Union, Dict, Any
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class ProstT5:
    def __init__(self, device: str = None):
        """
        Initialize ProstT5 model for protein sequence and structure analysis.
        
        Args:
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). 
                                  If None, will use CUDA if available.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get Hugging Face API token
        self.hf_token = os.getenv('HUGGINGFACE_API_TOKEN')
        if not self.hf_token:
            raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables. Please add it to your .env file.")
        
        # Load tokenizer with API token
        self.tokenizer = T5Tokenizer.from_pretrained(
            'Rostlab/ProstT5',
            do_lower_case=False,
            token=self.hf_token
        )
        
        # Load models with API token
        self.encoder_model = T5EncoderModel.from_pretrained(
            "Rostlab/ProstT5",
            token=self.hf_token
        ).to(self.device)
        
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(
            "Rostlab/ProstT5",
            token=self.hf_token
        ).to(self.device)
        
        # Set precision based on device
        if self.device == 'cpu':
            self.encoder_model.float()
            self.translation_model.float()
        else:
            self.encoder_model.half()
            self.translation_model.half()
            
        # Generation configurations
        self.gen_kwargs_aa2fold = {
            "do_sample": True,
            "num_beams": 3,
            "top_p": 0.95,
            "temperature": 1.2,
            "top_k": 6,
            "repetition_penalty": 1.2,
        }
        
        self.gen_kwargs_fold2aa = {
            "do_sample": True,
            "top_p": 0.85,
            "temperature": 1.0,
            "top_k": 3,
            "repetition_penalty": 1.2,
        }

    def _preprocess_sequences(self, sequences: List[str]) -> List[str]:
        """
        Preprocess protein sequences for the model.
        
        Args:
            sequences (List[str]): List of protein sequences
            
        Returns:
            List[str]: Preprocessed sequences
        """
        # Replace rare/ambiguous amino acids with X and add spaces between residues
        processed = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequences]
        return processed

    def get_embeddings(self, sequences: List[str], per_protein: bool = False) -> torch.Tensor:
        """
        Extract embeddings from protein sequences.
        
        Args:
            sequences (List[str]): List of protein sequences
            per_protein (bool): If True, return per-protein embeddings instead of per-residue
            
        Returns:
            torch.Tensor: Protein embeddings
        """
        # Preprocess sequences
        processed_seqs = self._preprocess_sequences(sequences)
        
        # Tokenize
        ids = self.tokenizer.batch_encode_plus(
            processed_seqs,
            add_special_tokens=True,
            padding="longest",
            return_tensors='pt'
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.encoder_model(
                ids.input_ids,
                attention_mask=ids.attention_mask
            ).last_hidden_state
        
        if per_protein:
            # Calculate mean embedding per protein
            embeddings = embeddings.mean(dim=1)
            
        return embeddings

    def sequence_to_structure(self, sequences: List[str]) -> List[str]:
        """
        Translate protein sequences to 3Di structure sequences.
        
        Args:
            sequences (List[str]): List of protein sequences
            
        Returns:
            List[str]: Predicted 3Di structure sequences
        """
        # Preprocess sequences
        processed_seqs = self._preprocess_sequences(sequences)
        
        # Add prefix for sequence to structure translation
        processed_seqs = ["<AA2fold> " + seq for seq in processed_seqs]
        
        # Tokenize
        ids = self.tokenizer.batch_encode_plus(
            processed_seqs,
            add_special_tokens=True,
            padding="longest",
            return_tensors='pt'
        ).to(self.device)
        
        # Generate structure sequences
        with torch.no_grad():
            translations = self.translation_model.generate(
                ids.input_ids,
                attention_mask=ids.attention_mask,
                max_length=ids.input_ids.shape[1],
                min_length=1,
                early_stopping=True,
                num_return_sequences=1,
                **self.gen_kwargs_aa2fold
            )
        
        # Decode and clean up
        decoded = self.tokenizer.batch_decode(translations, skip_special_tokens=True)
        structure_sequences = ["".join(ts.split(" ")) for ts in decoded]
        
        return structure_sequences

    def structure_to_sequence(self, structure_sequences: List[str]) -> List[str]:
        """
        Translate 3Di structure sequences to protein sequences.
        
        Args:
            structure_sequences (List[str]): List of 3Di structure sequences
            
        Returns:
            List[str]: Predicted protein sequences
        """
        # Preprocess sequences
        processed_seqs = self._preprocess_sequences(structure_sequences)
        
        # Add prefix for structure to sequence translation
        processed_seqs = ["<fold2AA> " + seq for seq in processed_seqs]
        
        # Tokenize
        ids = self.tokenizer.batch_encode_plus(
            processed_seqs,
            add_special_tokens=True,
            padding="longest",
            return_tensors='pt'
        ).to(self.device)
        
        # Generate protein sequences
        with torch.no_grad():
            translations = self.translation_model.generate(
                ids.input_ids,
                attention_mask=ids.attention_mask,
                max_length=ids.input_ids.shape[1],
                min_length=1,
                early_stopping=True,
                num_return_sequences=1,
                **self.gen_kwargs_fold2aa
            )
        
        # Decode and clean up
        decoded = self.tokenizer.batch_decode(translations, skip_special_tokens=True)
        protein_sequences = ["".join(ts.split(" ")) for ts in decoded]
        
        return protein_sequences
