"""
# Build `14-Nov-2024-15:07
Explanation:
EmbeddingLoader (Abstract Class):

Provides a blueprint for all embedding loaders to ensure a consistent interface.
Contains an abstract method load_embeddings().
GloVeEmbeddingLoader (Concrete Class):

Implements the EmbeddingLoader interface for loading GloVe embeddings.
Takes glove_path, vocab, and embedding_dim as arguments.
Error Handling:

Handles file not found (FileNotFoundError).
Handles invalid vector length or malformed lines in the GloVe file (ValueError).
Logs warnings for invalid lines and errors for unexpected issues using app_logger.
Logging:

Logs the number of successfully loaded words and issues during processing.
Clean and Modular:

Each responsibility is separated into cohesive, maintainable methods and classes.
Adheres to the Single Responsibility Principle by encapsulating the GloVe logic in a single class.
"""

import numpy as np
import sys, os
from abc import ABC, abstractmethod
from typing import Tuple, Dict

# Get the absolute path to the main directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from utils.logging_utils import app_logger 

class EmbeddingLoader(ABC):
    """
    Abstract base class for embedding loaders.

    Defines the interface for loading pre-trained embeddings.
    """

    @abstractmethod
    def load_embeddings(self) -> np.ndarray:
        """
        Abstract method for loading embeddings.

        Returns:
            np.ndarray: The embeddings matrix.
        """
        pass


class GloVeEmbeddingLoader(EmbeddingLoader):
    """
    Class to load GloVe embeddings from a file and map them to a given vocabulary.

    Args:
        glove_path (str): Path to the GloVe embeddings file.
        vocab (Dict[str, int]): Vocabulary dictionary mapping words to their indices.
        embedding_dim (int): Dimensionality of the embeddings.

    Raises:
        FileNotFoundError: If the GloVe file is not found.
        ValueError: If the GloVe file has an invalid format.

    Methods:
        load_embeddings(): Loads GloVe embeddings and maps them to the given vocabulary.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing:
            - The embeddings matrix (np.ndarray of shape [vocab_size, embedding_dim]).
            - Count of successfully matched words from the vocabulary.
    """

    def __init__(self, glove_path: str, vocab: Dict[str, int], embedding_dim: int = 100):
        self.glove_path = glove_path
        self.vocab = vocab
        self.embedding_dim = embedding_dim

    def load_embeddings(self) -> Tuple[np.ndarray, int]:
        """
        Loads GloVe embeddings and maps them to the provided vocabulary.

        Returns:
            Tuple[np.ndarray, int]: A tuple containing:
                - Embeddings matrix initialized with random values (shape: [vocab_size, embedding_dim]).
                - Count of successfully matched words from the vocabulary.
        
        Raises:
            FileNotFoundError: If the GloVe file is not found.
            ValueError: If the GloVe file has an invalid format.
        """
        try:
            # Initialize embeddings with random values
            embeddings = np.random.uniform(-0.25, 0.25, (len(self.vocab), self.embedding_dim))  # Random initialization
            matched_count = 0

            with open(self.glove_path, 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        values = line.split()
                        word = values[0]
                        vector = np.asarray(values[1:], dtype='float32')

                        if len(vector) != self.embedding_dim:
                            raise ValueError(f"Invalid vector length in GloVe file for word '{word}'.")

                        if word in self.vocab:
                            embeddings[self.vocab[word]] = vector
                            matched_count += 1
                    except ValueError as ve:
                        app_logger.warning(f"Skipping invalid line in GloVe file: {line.strip()}. Error: {ve}")
                    except Exception as e:
                        app_logger.error(f"Unexpected error while processing line: {line.strip()}. Error: {e}")

            app_logger.info(f"Loaded {matched_count}/{len(self.vocab)} words from GloVe embeddings.")
            return embeddings, matched_count

        except FileNotFoundError as fnf_error:
            app_logger.error(f"GloVe file not found at path: {self.glove_path}. Error: {fnf_error}")
            raise FileNotFoundError(f"GloVe file not found at path: {self.glove_path}") from fnf_error

        except Exception as e:
            app_logger.error(f"Failed to load GloVe embeddings. Error: {e}")
            raise RuntimeError("Error loading GloVe embeddings") from e

if __name__ == "__main__":
    from  utils.load_tokenizer import LoadTokenizerJson
    loading = LoadTokenizerJson()
    vocab = loading.load(path_tokenize="/workspaces/Spam-Email-Detection/models/vocab.json")
    
    # Path to GloVe embeddings
    glove_path = "/kaggle/input/glove6b100dtxt/glove.6B.100d.txt"
    glove_path = "/workspaces/Spam-Email-Detection/models/glove.6B.100d.txt"
    # Initialize the loader
    glove_loader = GloVeEmbeddingLoader(glove_path=glove_path, vocab=vocab, embedding_dim=100)

    # Load embeddings
    try:
        embeddings, matched_count = glove_loader.load_embeddings()
        print(f"Embeddings loaded successfully. Matched {matched_count} words.")
        print(f"Embeddings shape: {embeddings.shape}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Finsh Work. 14-Nov-2024-20:20
