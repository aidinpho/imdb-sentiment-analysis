# IMDb Sentiment Analysis: Machine Learning vs Deep Learning

A sentiment analysis project that compares two approaches to classifying IMDb movie reviews as **Positive** or **Negative**:

- **Machine Learning:** TF-IDF + Logistic Regression
- **Deep Learning:** PyTorch + LSTM

The repository contains two independent implementations of the same sentiment analysis task, allowing their architectures, evaluation results, and practical trade-offs to be explored.

## Overview

Sentiment analysis is a Natural Language Processing (NLP) task that identifies the emotional tone expressed in text.

This repository explores two approaches to binary sentiment classification. The traditional machine learning implementation converts reviews into TF-IDF features and uses Logistic Regression. The deep learning implementation converts tokens into embeddings and processes review sequences with an LSTM neural network.

Both implementations use the IMDb movie review dataset from Hugging Face:

`stanfordnlp/imdb`

The dataset contains 25,000 training reviews and 25,000 test reviews, with balanced positive and negative classes.

## Project Structure

```text
sentiment-analysis-deep-learning/
├── machine-learning/
│   ├── data/
│   ├── models/
│   │   └── sentiment_model.joblib
│   ├── notebooks/
│   ├── src/
│   │   ├── load_data.py
│   │   ├── train_model.py
│   │   └── predict.py
│   ├── tests/
│   ├── .gitignore
│   ├── README.md
│   └── requirements.txt
├── deep-learning/
│   ├── models/
│   │   ├── sentiment_lstm.pth
│   │   └── vocab.json
│   ├── notebooks/
│   │   └── 01_imdb_sentiment_analysis.ipynb
│   ├── src/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   ├── model.py
│   │   ├── preprocess.py
│   │   └── predict.py
│   ├── .gitignore
│   ├── README.md
│   └── requirements.txt
├── comparison/
│   ├── README.md
│   └── model_comparison.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

*Note: The comparison files are planned components and may not exist yet.*

## Approaches

### 1. Machine Learning

The machine learning implementation uses a scikit-learn pipeline consisting of:

- **TF-IDF:** Converts text into numerical features using unigrams and bigrams.
- **Logistic Regression:** Classifies the resulting feature vectors as positive or negative.
- **Joblib:** Saves the trained pipeline for inference without retraining.

Key configuration:

| Parameter | Value |
|---|---|
| Maximum TF-IDF features | 20,000 |
| N-gram range | `(1, 2)` |
| Classifier | Logistic Regression |
| Maximum iterations | 1,000 |
| Random state | 42 |

For implementation details, installation instructions, and evaluation results, see [`machine-learning/README.md`](machine-learning/README.md).

### 2. Deep Learning

The deep learning implementation uses a PyTorch LSTM network:

- **Embedding layer:** Maps tokens to 100-dimensional vectors.
- **LSTM layer:** Processes the review sequence with a hidden size of 64.
- **Linear classifier:** Produces scores for the two sentiment classes.
- **Packed sequences:** Avoids treating padding tokens as meaningful sequence input.

The vocabulary contains 30,000 tokens, and reviews are padded or truncated to a maximum length of 300 tokens.

The trained weights and vocabulary are saved separately, allowing predictions without retraining.

For implementation details, installation instructions, and evaluation results, see [`deep-learning/README.md`](deep-learning/README.md).

## Evaluation Results

The following results were reported by the existing project evaluations.

| Metric | Machine Learning | Deep Learning |
|---|---:|---:|
| Approach | TF-IDF + Logistic Regression | Embedding + LSTM |
| Test samples | 25,000 | 25,000 |
| Test accuracy | **88.30%** | 81.75% |
| Test loss | Not reported | 0.4455 |

The reported accuracy is higher for the traditional machine learning implementation. However, accuracy alone does not establish that one approach is universally better. A fair comparison should also consider preprocessing, training conditions, class-level metrics, inference speed, memory requirements, and reproducibility.

The reported results should be treated as results from the current model runs. Before drawing a definitive conclusion, verify that both evaluations used equivalent data splits and comparable evaluation procedures.

## Installation

### Prerequisites

- Python
- pip
- Internet access for the initial dataset download

Clone the repository:

```bash
git clone https://github.com/aidinpho/sentiment-analysis-deep-learning.git
cd sentiment-analysis-deep-learning
```

It is recommended to use a separate virtual environment for each implementation.

### Machine Learning

Install its dependencies from the repository root:

```bash
pip install -r machine-learning/requirements.txt
```

### Deep Learning

Install its dependencies from the repository root:

```bash
pip install -r deep-learning/requirements.txt
```

**PyTorch note:** The appropriate PyTorch installation depends on your operating system and whether you intend to use a CPU or a compatible CUDA-enabled GPU.

## Usage

Run the following commands from the repository root.

### Machine Learning

**Train the model:**

```bash
python machine-learning/src/train_model.py
```

**Predict sentiment:**

```bash
python machine-learning/src/predict.py
```

The prediction script loads the saved TF-IDF and Logistic Regression pipeline and asks for a movie review.

### Deep Learning

**Predict sentiment using the saved model:**

```bash
python deep-learning/src/predict.py
```

The script loads the saved LSTM weights and vocabulary, preprocesses the review, and predicts its sentiment.

For training and evaluation details, follow the notebook in `deep-learning/notebooks/`.

## Comparison

The `comparison/` directory is intended to contain a notebook comparing the two approaches in a consistent evaluation workflow.

Potential comparison dimensions include:

- Test accuracy and class-level precision, recall, and F1-score
- Confusion matrices
- Training time
- Inference latency
- Model size and memory requirements
- Strengths and limitations of each approach

The comparison notebook should be completed using actual measurements rather than assumed results.

## Limitations

- Both models perform binary sentiment classification.
- TF-IDF does not fully capture word order or complex contextual relationships.
- LSTM predictions can still be affected by vocabulary coverage, sequence length, and preprocessing choices.
- Performance on other datasets may differ from the reported IMDb results.
- The reported metrics should be verified under a consistent evaluation procedure before making definitive comparisons.

## Future Improvements

- Complete the model comparison notebook.
- Add automated tests for preprocessing and inference.
- Compare inference speed and model size.
- Evaluate both approaches using the same test examples.
- Experiment with alternative classifiers and neural architectures.
- Improve reproducibility by documenting software versions and random seeds.

## License

No license has been specified yet. Add a `LICENSE` file if you intend to distribute the project under an explicit open-source license.