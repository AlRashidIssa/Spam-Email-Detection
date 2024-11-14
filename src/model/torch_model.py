import torch
import torch.nn as nn
import os
import sys

# Get the absolute path to the main directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from utils.logging_utils import app_logger


class SpamClassifierModel(nn.Module):
    """
    Spam Classification Model using pre-trained embeddings, LSTM, and a fully connected layer.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimensionality of the input embeddings.
        embedding_matrix (torch.Tensor): Pre-trained embedding matrix (e.g., GloVe).
        hidden_dim (int): Dimensionality of the LSTM's hidden states.
        output_dim (int): Number of output classes.

    Methods:
        forward(x): Computes the class probabilities for input sequences.

    Returns:
        torch.Tensor: Class probabilities for each input (shape: [batch_size, output_dim]).
    """
    def __init__(self, vocab_size: int, embedding_dim: int, embedding_matrix: torch.Tensor,
                 hidden_dim: int, output_dim: int):
        super(SpamClassifierModel, self).__init__()
        try:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
            self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, 
                                num_layers=1, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)  # hidden_dim * 2 for bidirectional
            self.softmax = nn.Softmax(dim=1)
            app_logger.info("Successfully initialized SpamClassifierModel")
        except Exception as e:
            app_logger.error(f"Error initializing SpamClassifierModel: {e}")
            raise ValueError("Failed to initialize SpamClassifierModel") from e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SpamClassifierModel.

        Args:
            x (torch.Tensor): Input tensor containing token indices (shape: [batch_size, sequence_length]).
        
        Raises:
            RuntimeError: If the forward pass fails.

        Returns:
            torch.Tensor: Class probabilities for each input (shape: [batch_size, output_dim]).
        """
        try:
            embedded = self.embedding(x)  # Pass through embedding layer
            _, (hidden, _) = self.lstm(embedded)  # Pass through LSTM
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # Concatenate forward/backward states
            output = self.fc(hidden)  # Pass through fully connected layer
            return self.softmax(output)  # Apply softmax
        except Exception as e:
            app_logger.error(f"Error in SpamClassifierModel forward pass: {e}")
            raise RuntimeError("SpamClassifierModel forward pass failed") from e


if __name__ == "__main__":
    try:
        # Define sample parameters
        vocab_size = 10000  # Vocabulary size (not used directly as embeddings are pre-trained)
        embedding_dim = 300  # Dimension of word embeddings
        hidden_dim = 128  # LSTM hidden state dimension
        output_dim = 2  # Number of output classes (e.g., Spam or Not Spam)

        # Create a random embedding matrix for demonstration purposes
        embedding_matrix = torch.randn(vocab_size, embedding_dim)

        # Initialize the model
        model = SpamClassifierModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            embedding_matrix=embedding_matrix,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        app_logger.info("Model initialized successfully!")

        # Create a sample batch of input sequences
        batch_size = 4
        sequence_length = 10
        sample_input = torch.randint(0, vocab_size, (batch_size, sequence_length))

        # Perform forward pass
        with torch.no_grad():  # Disable gradient computation for inference
            output = model(sample_input)

        print(f"Sample input:\n{sample_input}")
        print(f"Model output probabilities:\n{output}")
        print(f"Predicted classes:\n{torch.argmax(output, dim=1)}")

    except Exception as e:
        app_logger.error(f"An error occurred during main execution: {e}")