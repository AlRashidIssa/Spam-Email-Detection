import os
import sys
import torch
import numpy as np
import  torch.nn as nn
from typing import Any, Dict
from abc import ABC, abstractmethod

# Adjusting the main directory for imports
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

# Import custom modules
from model.embedding import GloVeEmbeddingLoader
from model.torch_model import SpamClassifierModel
from utils.load_tokenizer import LoadTokenizerJson
from utils.logging_utils import app_logger


class ILoadTorchModel(ABC):
    """
    Abstract Interface for loading the spam classifier model and its components.
    """

    @abstractmethod
    def load(self, model_path: str, glove_path: str, vocab: Dict[str, int],
             embedding_dim: int, hidden_dim: int, output_dim: int) -> torch.nn.Module:
        """
        Abstract method to load the spam classifier model and its components.

        Args:
            model_path (str): Path to the saved model file.
            glove_path (str): Path to the GloVe embeddings file.
            vocab (dict): Vocabulary mapping words to indices.
            embedding_dim (int): Dimensionality of word embeddings.
            hidden_dim (int): LSTM hidden state dimensionality.
            output_dim (int): Number of output classes.

        Returns:
            model (SpamClassifierModel): Loaded model ready for inference.
        """
        pass


class LoadTorchModel(ILoadTorchModel):
    """
    Implementation for loading the spam classifier model and its components.
    """

    def load(self, model_path: str, glove_path: str, vocab: Dict[str, int],
             embedding_dim: int, hidden_dim: int, output_dim: int) -> torch.nn.Module:
        """
        Load the spam classifier model and its components.

        Args:
            model_path (str): Path to the saved model file.
            glove_path (str): Path to the GloVe embeddings file.
            vocab (dict): Vocabulary mapping words to indices.
            embedding_dim (int): Dimensionality of word embeddings.
            hidden_dim (int): LSTM hidden state dimensionality.
            output_dim (int): Number of output classes.

        Returns:
            model (SpamClassifierModel): Loaded model ready for inference.

        Raises:
            FileNotFoundError: If the model file or GloVe file is not found.
            ValueError: If there is an issue with the dimensions or data formats.
            Exception: For any other unforeseen errors during the process.
        """
        try:
            # Step 1: Validate file paths
            if not os.path.exists(model_path):
                app_logger.error(f"Model file not found at {model_path}.")
                raise FileNotFoundError(f"Model file not found at {model_path}.")
            if not os.path.exists(glove_path):
                app_logger.error(f"GloVe embeddings file not found at {glove_path}.")
                raise FileNotFoundError(f"GloVe embeddings file not found at {glove_path}.")

            app_logger.info("Validated file paths successfully.")

            # Step 2: Load GloVe embeddings
            app_logger.info("Loading GloVe embeddings...")
            glove_loader = GloVeEmbeddingLoader(glove_path, vocab, embedding_dim)
            glove_embeddings, _ = glove_loader.load_embeddings()
            embedding_matrix = torch.tensor(glove_embeddings, dtype=torch.float)
            app_logger.info("GloVe embeddings loaded successfully.")

            # Step 3: Initialize the model
            app_logger.info("Initializing the model...")
            # Initialize the model
            model = SpamClassifierModel(
                vocab_size=len(vocab),
                embedding_dim=embedding_dim,
                embedding_matrix=embedding_matrix,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )
            app_logger.info("Model initialized successfully.")

            # Step 4: Load saved model weights
            app_logger.info("Loading model weights...")
            state_dict = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))

            # Remove `module.` prefix if present
            if any(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

            model.load_state_dict(state_dict)
            model.eval()
            app_logger.info("Model weights loaded successfully.")
            return model
        except FileNotFoundError as fnf_error:
            app_logger.error(f"FileNotFoundError: {fnf_error}")
            raise
        except ValueError as ve:
            app_logger.error(f"ValueError: {ve}")
            raise
        except Exception as e:
            app_logger.error(f"An unexpected error occurred: {e}")
            raise


if __name__ == "__main__":
    try:
        # Define paths and parameters
        model_path = "/workspaces/Spam-Email-Detection/models/spam_classifier_model.pth"
        glove_path = "/workspaces/Spam-Email-Detection/models/glove.6B.100d.txt"
        tokenizer_path = "/workspaces/Spam-Email-Detection/models/vocab.json"

        # Load vocabulary using tokenizer utility
        app_logger.info("Loading vocabulary...")
        vocab = LoadTokenizerJson().load(tokenizer_path)
        app_logger.info("Vocabulary loaded successfully.")

        # Define other model parameters
        embedding_dim = 100  # GloVe embedding dimension
        hidden_dim = 128  # Hidden state size
        output_dim = 2  # Number of classes (Spam, Not Spam)

        # Load the model
        app_logger.info("Loading the model pipeline...")
        model_loader = LoadTorchModel()
        model = model_loader.load(model_path=model_path,
                                  glove_path=glove_path,
                                  vocab=vocab,
                                  embedding_dim=embedding_dim,
                                  hidden_dim=hidden_dim,
                                  output_dim=output_dim)
        app_logger.info("Successfull Load Model.")

    except FileNotFoundError as fnf_error:
        app_logger.error(f"FileNotFoundError: {fnf_error}")
    except Exception as e:
        app_logger.error(f"An unexpected error occurred in the main pipeline: {e}")
        print(f"An error occurred: {e}")
