import os
import sys
import torch
from abc import ABC, abstractmethod
from typing import Dict

# Add the main directory to the system path for module imports
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from utils.logging_utils import app_logger


class IPredictionAPI(ABC):
    """
    Interface for Prediction API.
    """

    @abstractmethod
    def predict(self, tensor_text: torch.Tensor, model: torch.nn.Module) -> str:
        """
        Abstract method for performing predictions.

        Args:
            tensor_text (torch.Tensor): Preprocessed input tensor.
            model (torch.nn.Module): Trained PyTorch model.

        Returns:
            str: Prediction result ("Ham" or "Spam").
        """
        pass


class PredictionAPI(IPredictionAPI):
    """
    Implementation of the Prediction API.
    """

    def predict(self, tensor_text: torch.Tensor, model: torch.nn.Module) -> str:
        """
        Perform prediction using the trained model.

        Args:
            tensor_text (torch.Tensor): Preprocessed input tensor.
            model (torch.nn.Module): Trained PyTorch model.

        Returns:
            str: Prediction result ("Ham" or "Spam").
        """
        try:
            app_logger.info("Starting prediction process.")
            # Perform prediction
            with torch.no_grad():
                output = model(tensor_text)
                confidence = torch.nn.functional.softmax(output, dim=1).max().item()
                _, predicted = torch.max(output, 1)
                result = "Non-Spam" if predicted.item() == 0 else "Spam"
                app_logger.info(f"Prediction: {result}, Confidence: {confidence:.2f}")

            return result, confidence
        except Exception as e:
            app_logger.error(f"Error during prediction: {e}")
            raise RuntimeError("Failed to perform prediction") from e



if __name__ == "__main__":
    try:
        # Load the model and vocabulary
        app_logger.info("Loading model and vocabulary.")
        glove_path = "/workspaces/Spam-Email-Detection/models/glove.6B.100d.txt"
        model_path = "/workspaces/Spam-Email-Detection/models/spam_classifier_model.pth"
        vocab_path = "/workspaces/Spam-Email-Detection/models/vocab.json"
        embedding_dim = 100
        hidden_dim = 128
        output_dim = 2
        from model.load_model import LoadTorchModel
        from preprocessing.preprocessing import CleanText
        from utils.load_tokenizer import LoadTokenizerJson
        vocab = LoadTokenizerJson().load(path_tokenize=vocab_path)
        clean_tensor = CleanText().clean(text="Hello AlRashid how are u",
                                         vocab=vocab)
        model = LoadTorchModel().load(model_path=model_path,
                                      glove_path=glove_path,
                                      vocab=vocab,
                                      embedding_dim=embedding_dim,
                                      hidden_dim=hidden_dim,
                                      output_dim=output_dim)
        massege = PredictionAPI().predict(tensor_text=clean_tensor,
                                          model=model)
        print(massege)
    except FileNotFoundError as fnf_error:
        app_logger.error(f"FileNotFoundError: {fnf_error}")
        print(f"Error: {fnf_error}")
    except Exception as e:
        app_logger.error(f"Unexpected error: {e}")
        print(f"An error occurred: {e}")
# Finsh Work. 14-Nov-2024-20:20
