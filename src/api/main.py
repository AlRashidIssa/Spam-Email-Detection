import os
import sys
from abc import ABC, abstractmethod
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

# Import custom utilities
from utils.logging_utils import app_logger
from model.load_model import LoadTorchModel
from utils.load_tokenizer import LoadTokenizerJson
from preprocessing.preprocessing import CleanText
from prediction.prediction import PredictionAPI

# Initialize FastAPI and templates
APP = FastAPI()
TEMPLATES = Jinja2Templates(directory="/workspaces/Spam-Email-Detection/src/api/templates")

# Load model and vocabulary with logging and error handling
try:
    app_logger.info("Loading vocabulary...")
    vocab_path = "/workspaces/Spam-Email-Detection/models/vocab.json"
    vocab = LoadTokenizerJson().load(path_tokenize=vocab_path)
    app_logger.info("Vocabulary loaded successfully.")

    app_logger.info("Loading model...")
    glove_path = "/workspaces/Spam-Email-Detection/models/glove.6B.100d.txt"
    model_path = "/workspaces/Spam-Email-Detection/models/spam_classifier_model.pth"
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 2
    MODEL = LoadTorchModel().load(
        model_path=model_path,
        glove_path=glove_path,
        vocab=vocab,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )
    app_logger.info("Model loaded successfully.")
except Exception as e:
    app_logger.error(f"Error initializing model or vocabulary: {e}")
    raise RuntimeError("Failed to initialize the model or vocabulary.") from e


@APP.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    """
    GET method to render the HTML form.
    """
    app_logger.info("Rendering HTML form.")
    try:
        return TEMPLATES.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        app_logger.error(f"Error rendering form: {e}")
        raise HTTPException(status_code=500, detail="Unable to render the form.")


@APP.post("/predict", response_class=HTMLResponse)
async def predict_spam(request: Request, text: str = Form(...)):
    """
    POST method to handle spam email detection.

    Args:
        request (Request): The FastAPI request object.
        text (str): The input text to analyze.

    Returns:
        HTMLResponse: The rendered HTML response with prediction results or error messages.
    """
    app_logger.info("Received text for prediction.")
    try:
        # Clean and preprocess the input text
        app_logger.info("Cleaning and preprocessing input text.")
        clean_tensor = CleanText().clean(text=text, vocab=vocab)

        # Perform prediction
        app_logger.info("Running model prediction.")
        prediction_result, confidence = PredictionAPI().predict(
            model=MODEL, tensor_text=clean_tensor
        )

        app_logger.info(f"Prediction: {prediction_result}, Confidence: {confidence:.2f}")
        return TEMPLATES.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": prediction_result,
                "input_text": text,
                "confidence": f"{confidence * 100:.2f}%",  # Convert to percentage
            },
        )

    except RuntimeError as e:
        app_logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again.")
    except Exception as e:
        app_logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

