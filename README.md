# Spam Email Detection

## Project Overview

The **Spam Email Detection** project is a machine learning model that classifies emails into two categories: **Spam** and **Ham** (non-spam). This model uses **PyTorch** for model training, **NLTK** for text preprocessing, and other useful libraries to preprocess and classify email content.

### Technologies Used:
- **PyTorch**: For model building and training.
- **NLTK**: For text preprocessing, including tokenization, stop word removal, and more.
- **FastAPI**: To serve the model through an API.
- **Scikit-learn**: For model evaluation and metrics like precision, recall, and F1-score.
- **Git LFS**: For managing large files such as model weights.

## API Documentation

The API exposes a `POST` endpoint that allows users to classify email content as **Spam** or **Ham**.

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. **POST /predict**
This endpoint accepts the content of an email and returns the prediction on whether the email is spam or ham.

**Request Body Example:**
```json
{
  "email_content": "Congratulations! You've won a free vacation. Click here to claim your prize."
}
```

**Response Example:**
```json
{
  "prediction": "Spam"
}
```

### How to Run the API

#### Step 1: Clone the Repository
Clone the project to your local machine by running the following command in your terminal:
```bash
git clone https://github.com/AlRashidIssa/Spam-Email-Detection.git
cd Spam-Email-Detection
```

#### Step 2: Install Dependencies
Install the required dependencies by running:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Step 3: Run the FastAPI Server
Start the API server using **Uvicorn**:
```bash
uvicorn main:app --reload
```

The API will be available locally at `http://localhost:8000`.

#### Step 4: Testing the API
You can test the API by sending a `POST` request to the `/predict` endpoint. Tools like **Postman** or **curl** can be used to send the request.

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
  "email_content": "Congratulations! You've won a free vacation. Click here to claim your prize."
}'
```

The response will contain the classification result (`Spam` or `Ham`).

## Model Evaluation

The model was evaluated on a test dataset and produced the following results:

### Test Performance:
- **Test Loss:** 0.3568
- **Test Accuracy:** 95.68%

### Classification Report:

```
              precision    recall  f1-score   support

         Ham       0.94      0.98      0.96      3309
        Spam       0.98      0.94      0.96      3434

    accuracy                           0.96      6743
   macro avg       0.96      0.96      0.96      6743
weighted avg       0.96      0.96      0.96      6743
```

### Key Model Metrics:
- **Precision**: The model is highly precise, correctly identifying both **Spam** and **Ham** emails.
- **Recall**: The model achieves good recall for both classes, with slightly better recall for **Ham** emails.
- **F1-score**: The model has an excellent F1-score for both categories, showing balance between precision and recall.
- **Accuracy**: The model achieved an impressive accuracy of 95.68% on the test data, demonstrating its effectiveness in classifying emails as **Spam** or **Ham**.

## Project Structure

Here’s a brief overview of the project’s structure:

```
Spam-Email-Detection/
│
├── main.py                # FastAPI application
├── models/                # Folder containing the trained model and related files
│   └── spam_classifier_model.pth  # The trained PyTorch model
├── requirements.txt       # Python dependencies for the project
├── .gitignore             # Files and folders to ignore in Git
└── README.md              # This file
```

## Notes:
- **Text Preprocessing**: The model utilizes **NLTK** for preprocessing tasks like tokenization, removing stop words, and lemmatization.
- **Model Training**: The model is built using **PyTorch** and trained on a dataset of labeled emails.
- **File Handling**: If you're using GitHub, you may encounter file size limitations for large files like the trained model weights. You may need to use **Git Large File Storage (LFS)** or host the model file externally.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements
- [FastAPI](https://fastapi.tiangolo.com/) for building the API.
- [PyTorch](https://pytorch.org/) for building and training the machine learning model.
- [NLTK](https://www.nltk.org/) for text preprocessing.
