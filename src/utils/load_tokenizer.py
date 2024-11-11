r"""\
    Logging and Error Handling:

    The load() method now correctly logs different errors at each step, including invalid file paths, invalid JSON, and empty files.
    Exception handling is done using try-except blocks, which provide specific handling for different error types (ValueError, FileNotFoundError, TypeError, JSONDecodeError).
    Exception Types:

    TypeError is raised if the path_tokenize is not a string.
    FileNotFoundError is raised if the specified file path does not exist.
    ValueError is raised if the file cannot be decoded or is empty.
    JSONDecodeError is specifically caught to handle issues with parsing the JSON data.
    Method Documentation:

    Added detailed docstrings for both the abstract class and the concrete implementation. This helps clarify what each method does, the expected parameter types, and the return values.
    Return Data Type:

    The method now returns a dictionary (dict) containing the loaded tokenizer data. You can adjust this if your tokenizer format is more complex (e.g., if it contains additional structures like vocab files).
"""
import os, sys, json
# Get the absolute path to direcoty one level abovce the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)
from abc import ABC, abstractmethod
from utils.logging_utils import app_logger

class ILoadTokenizerJson(ABC):
    """
    Abstract class for loading a tokenizer from a JSON file.

    Methods
    -------
    load(path_tokenize: str) -> dict:
        Loads the tokenizer data from a JSON file.
    """
    @abstractmethod
    def load(self, path_tokenize: str) -> dict:
        """
        Abstract method to load the tokenizer from a JSON file.

        Parameters:
        path_tokenize (str): The path to the JSON file containing the tokenizer.

        Returns:
        dict: The tokenizer data loaded from the file.
        """
        pass

class LoadTokenizerJson(ILoadTokenizerJson):
    """
    Concrete implementation of ILoadTokenizerJson to load a tokenizer from a JSON file.

    Methods
    -------
    load(path_tokenize: str) -> dict:
        Loads the tokenizer data from the provided JSON file and returns it.
    """
    def load(self, path_tokenize: str) -> dict:
        """
        Loads the tokenizer data from a JSON file.

        Parameters:
        path_tokenize (str): The path to the JSON file containing the tokenizer.

        Returns:
        dict: The tokenizer data loaded from the file.

        Raises:
        ValueError: If the file does not exist or cannot be loaded.
        TypeError: If the path_tokenize parameter is not a string.
        """
        # Ensure the provided path is a string
        if not isinstance(path_tokenize, str):
            app_logger.error("The provided path must be a string.")
            raise TypeError("The path_tokenize parameter must be a string.")

        # Check if the file exists
        if not os.path.exists(path_tokenize):
            app_logger.error(f"Tokenizer file not found at the path: {path_tokenize}")
            raise FileNotFoundError(f"Tokenizer file not found at the path: {path_tokenize}")

        # Try to load the JSON data
        try:
            with open(path_tokenize, 'r') as f:
                tokenizer_data = json.load(f)
                
                # Check if the tokenizer data is empty
                if tokenizer_data is None:
                    app_logger.error(f"The tokenizer file at {path_tokenize} is empty.")
                    raise ValueError("The tokenizer file is empty.")
                
                app_logger.info(f"Successfully loaded tokenizer from {path_tokenize}")
                return tokenizer_data

        except json.JSONDecodeError:
            app_logger.error(f"Error decoding JSON from the file: {path_tokenize}")
            raise ValueError(f"Error decoding JSON from the file: {path_tokenize}")
        
        except Exception as e:
            app_logger.error(f"An unexpected error occurred while loading the tokenizer: {str(e)}")
            raise RuntimeError(f"An unexpected error occurred: {str(e)}")

# Test Case
if __name__ == "__main__":
    loading = LoadTokenizerJson()
    tokenizer = loading.load(path_tokenize="/workspaces/Spam-Email-Detection/models/vocab.json")
    print(tokenizer)