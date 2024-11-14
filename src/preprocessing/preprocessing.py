import os, sys, re, nltk
import torch
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../")) # Adjusting the main directory for imports
sys.path.append(MAIN_DIR)
from abc import ABC, abstractmethod
from typing import Any, Union
from nltk.corpus import stopwords
from utils.logging_utils import app_logger
from utils.load_tokenizer import LoadTokenizerJson


# Ensure NLTK dependencies are downloaded
nltk.download('stopwords')
nltk.download('punkt')

STOP_WORDS = set(stopwords.words("english"))
MAX_SEQ_LENTH = 1000
class ICleanText(ABC):
    """
    Interface for cleaning text data.
    """

    @abstractmethod
    def clean(self, text: str) -> Union[str, None]:
        """
        Abstract method to clean the input text by removing unwanted characters and stopwords.

        Args:
            text (str): The text to be cleaned.

        Returns:
            Union[str, None]: The cleaned text string or None if cleaning fails.
        """
        pass

class CleanText(ICleanText):
    """
    Concrete implementation of the ICleanText interface for cleaning text data.
    """

    def clean(self, text: str, vocab: dict) -> Union[torch.Tensor, None]:
        """
        Cleans the input text by removing special characters, numbers, and stopwords.

        Args:
            text (str): The text to be cleaned.

        Returns:
            Union[str, None]: The cleaned text string if successful, or None if an error occurs.

        Raises:
            ValueError: If the input text is not a string.
        """
        try:
            if not isinstance(text, str):
                raise ValueError("Input text must be a string.")
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-z\s]', '', text)
            
            # Remove stopwords
            words = text.split()
            words = [word for word in words if word not in STOP_WORDS]

            indices = [vocab.get(word, vocab['<UNK>']) for word in words]

            # Pad or truncate to the specified max sequence length
            if len(indices) < MAX_SEQ_LENTH:
                indices.extend([0] * (MAX_SEQ_LENTH - len(indices)))
            else:
                indices = indices[:MAX_SEQ_LENTH]
            # Join the words back into a single string
            app_logger.info("Clearing Text!")
            return torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # Adding batch dimension

        
        except ValueError as ve:
            app_logger.error(f"ValueError: {ve}")
            return None
        except Exception as e:
            app_logger.error(f"An error occurred during text cleaning: {e}")
            return None
        

if __name__ == "__main__":
    tokenizer_path = "/workspaces/Spam-Email-Detection/models/vocab.json"
    vocab = LoadTokenizerJson().load(tokenizer_path)
    iclearn = CleanText()
    clearn_text = iclearn.clean(text="@@333Hello worlds .///#4 def call inging.. !!! I Love Salah and Naseerasdfghjkl",
                                vocab=vocab)
    print("Clearning Text", clearn_text)