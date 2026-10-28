# 🎬 IMDb Sentiment Analysis | Machine Learning

A machine learning project that classifies IMDb movie reviews as **Positive** or **Negative** using Natural Language Processing (NLP), TF-IDF feature extraction, and Logistic Regression.

This project demonstrates a complete text classification pipeline, from dataset preparation and text preprocessing to model training, evaluation, and inference.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technologies](#-technologies)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Evaluation Results](#-evaluation-results)
- [Confusion Matrix](#-confusion-matrix)
- [How It Works](#-how-it-works)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Related Project](#-related-project)
- [License](#-license)

---

## 🎯 Overview

Sentiment analysis is a Natural Language Processing task that identifies the emotional tone expressed in a piece of text.

In this project, a machine learning pipeline analyzes movie reviews and predicts whether each review expresses a positive or negative sentiment.

The project uses **TF-IDF** to transform text into numerical features and **Logistic Regression** to perform binary classification.

The trained model is saved with Joblib, allowing predictions to be made without retraining the model every time.

---

## ✨ Features

- IMDb movie review sentiment classification
- Text cleaning and normalization
- TF-IDF feature extraction
- Logistic Regression classification
- Automated model training
- Evaluation using accuracy, precision, recall, and F1-score
- Confusion matrix generation
- Model persistence with Joblib
- Interactive command-line predictions
- Separate scripts for data preparation, training, and inference

---

## 🛠️ Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| pandas | Data manipulation |
| scikit-learn | Feature extraction, classification, and evaluation |
| Hugging Face Datasets | Loading the IMDb dataset |
| TF-IDF | Converting text into numerical features |
| Logistic Regression | Sentiment classification |
| Joblib | Saving and loading the trained model |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Jupyter Notebook | Interactive data exploration |

---

## 📚 Dataset

This project uses the IMDb movie review dataset available through Hugging Face Datasets.

**Dataset identifier:** `stanfordnlp/imdb`

The dataset contains 50,000 movie reviews divided into training and test sets.

| Split | Number of Reviews |
|---|---:|
| Training set | 25,000 |
| Test set | 25,000 |
| Total | 50,000 |

Each review has a binary sentiment label:

| Label | Sentiment |
|---|---|
| `0` | Negative |
| `1` | Positive |

The dataset is loaded automatically by the training script. An internet connection may be required when downloading it for the first time.

---

## 🧠 Model Architecture

The project uses a traditional machine learning pipeline composed of two main components.

### 1. TF-IDF Vectorizer

The `TfidfVectorizer` converts text reviews into numerical feature vectors based on the importance of words and word combinations.

Configuration:

| Parameter | Value |
|---|---|
| `max_features` | 20,000 |
| `ngram_range` | `(1, 2)` |
| `stop_words` | `english` |

Using unigrams and bigrams allows the model to consider individual words as well as two-word combinations.

### 2. Logistic Regression

Logistic Regression learns to distinguish positive and negative reviews using the TF-IDF features.

Configuration:

| Parameter | Value |
|---|---|
| `max_iter` | 1,000 |
| `random_state` | 42 |

Both components are combined into a scikit-learn `Pipeline`, ensuring that text transformation and classification are applied consistently during training and prediction.

---

## 📁 Project Structure

```text
machine-learning/
├── models/
│   └── sentiment_model.joblib
├── notebooks/
├── src/
│   ├── load_data.py
│   ├── train_model.py
│   └── predict.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

### Main Files

| File | Description |
|---|---|
| `src/load_data.py` | Loads the IMDb dataset, explores label distributions, cleans text, and prepares training and test data |
| `src/train_model.py` | Builds the TF-IDF and Logistic Regression pipeline, trains the model, evaluates performance, and saves the model |
| `src/predict.py` | Loads the saved model and predicts the sentiment of a user-provided review |
| `models/sentiment_model.joblib` | Serialized trained model |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or a compatible Python version
- pip
- Internet access for the initial dataset download

### 1. Clone the Repository

If you are cloning the combined repository:

```bash
git clone https://github.com/aidinpho/sentiment-analysis-deep-learning.git
cd sentiment-analysis-deep-learning
```

### 2. Install Dependencies

From the repository root, run:

```bash
pip install -r machine-learning/requirements.txt
```

Alternatively, create and activate a dedicated virtual environment for this project.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r machine-learning/requirements.txt
```

If you already have an active environment with the required dependencies installed, you do not need to create another one.

---

## 🚀 Usage

Run the following commands from the repository root.

### 1. Train the Model

```bash
python machine-learning/src/train_model.py
```

The script will:

1. Load the IMDb dataset.
2. Clean and prepare the reviews.
3. Create the TF-IDF and Logistic Regression pipeline.
4. Train the model.
5. Evaluate performance on the test set.
6. Save the trained model to `machine-learning/models/sentiment_model.joblib`.

**Note:** Training is not required if the saved model is already available and you only want to make predictions.

### 2. Predict Sentiment

```bash
python machine-learning/src/predict.py
```

Enter a movie review when prompted.

Example input:

```text
This movie was amazing and wonderful. I really enjoyed it.
```

Expected output:

```text
Predicted sentiment: Positive
```

The output depends on the text entered by the user.

---

## 📊 Evaluation Results

The model was evaluated on the IMDb test set containing 25,000 reviews.

### Overall Performance

| Metric | Result |
|---|---:|
| Training samples | 25,000 |
| Test samples | 25,000 |
| Correct predictions | 22,075 |
| Incorrect predictions | 2,925 |
| Test accuracy | **88.30%** |

The model correctly classified 22,075 out of 25,000 test reviews.

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.89 | 0.88 | 0.88 | 12,500 |
| Positive | 0.88 | 0.89 | 0.88 | 12,500 |
| Macro Average | 0.88 | 0.88 | 0.88 | 25,000 |
| Weighted Average | 0.88 | 0.88 | 0.88 | 25,000 |

The model achieves balanced performance across positive and negative reviews, with an overall test accuracy of **88.30%**.

---

## 🔍 Confusion Matrix

The confusion matrix produced during evaluation is:

```text
[[11001  1499]
 [ 1426 11074]]
```

Rows represent the actual labels, while columns represent the predicted labels. The label order is Negative (`0`) followed by Positive (`1`).

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Actual Negative | 11,001 | 1,499 |
| Actual Positive | 1,426 | 11,074 |

### Interpretation

- **11,001** negative reviews were correctly classified.
- **1,499** negative reviews were incorrectly classified as positive.
- **1,426** positive reviews were incorrectly classified as negative.
- **11,074** positive reviews were correctly classified.

These results indicate that the model performs similarly on both sentiment classes, although some reviews remain difficult to classify correctly.

---

## 🔄 How It Works

The workflow follows these steps:

1. **Dataset loading:** Download or load the IMDb dataset using Hugging Face Datasets.
2. **Text preprocessing:** Remove HTML line-break tags, convert text to lowercase, normalize whitespace, and trim surrounding spaces.
3. **Feature extraction:** Convert review text into TF-IDF features using unigrams and bigrams.
4. **Model training:** Train a Logistic Regression classifier on the training data.
5. **Model evaluation:** Measure accuracy and generate a classification report and confusion matrix using the test data.
6. **Model persistence:** Save the trained pipeline with Joblib.
7. **Inference:** Load the saved pipeline, preprocess a new review, and predict its sentiment.

---

## ⚠️ Limitations

- The model performs binary classification and does not predict sentiment intensity.
- TF-IDF features do not fully capture context, sarcasm, or complex language relationships.
- Predictions may be less reliable for reviews that differ substantially from the IMDb training data.
- The reported performance reflects the specified IMDb test set and does not guarantee the same accuracy on other datasets.
- The project currently has no automated test suite.

---

## 🔮 Future Improvements

Potential improvements include:

- Add unit tests for text preprocessing and prediction.
- Add exploratory data analysis and visualizations.
- Experiment with TF-IDF parameters and alternative classifiers.
- Perform systematic hyperparameter tuning.
- Evaluate the model on additional datasets.
- Add confidence scores to predictions.
- Compare the model with a deep learning implementation using an LSTM network.
- Create a unified comparison notebook to evaluate both approaches.

---

## 🔗 Related Project

This project is part of a broader sentiment analysis repository that also includes a deep learning implementation.

The goal is to compare traditional machine learning methods with neural-network-based approaches on the same sentiment analysis task.

See the repository's main README for an overview of the combined project.

---

## 📄 License

A license has not yet been specified for this project. Add a `LICENSE` file before redistributing the code under an explicit open-source license.