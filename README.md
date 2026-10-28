# IMDb Sentiment Analysis with PyTorch

A deep learning project that classifies IMDb movie reviews as **positive** or **negative** using an Embedding layer and an LSTM neural network built with PyTorch.

## Overview

This project uses a Long Short-Term Memory (LSTM) network to perform sentiment analysis on movie reviews.

Given a review, the model converts its words into numerical token IDs, maps them to dense vectors using an embedding layer, processes the sequence with an LSTM, and predicts one of two sentiment classes:

- **Negative (0)**
- **Positive (1)**

The trained model weights and vocabulary are saved so that predictions can be made without retraining the network.

## Technologies

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- NLTK
- Hugging Face Datasets
- Matplotlib
- Seaborn
- Jupyter Notebook

## Dataset

This project uses the IMDb movie review dataset from Hugging Face:

`stanfordnlp/imdb`

Dataset statistics:

- Training samples: 25,000
- Test samples: 25,000
- Number of classes: 2
- Class distribution: Balanced between positive and negative reviews

## Model Architecture

The model consists of three main components:

1. **Embedding Layer:** Maps token IDs to 100-dimensional vectors.
2. **LSTM Layer:** Processes the review sequence with a hidden size of 64.
3. **Linear Classifier:** Produces two output scores, one for each sentiment class.

The model uses packed sequences to avoid processing padding tokens as meaningful input.

### Input Processing

- HTML tags are removed, and whitespace is normalized.
- Reviews are tokenized and converted to lowercase.
- The vocabulary contains 30,000 tokens, including padding and unknown-token entries.
- Each review is truncated or padded to a maximum length of 300 tokens.
- Padding tokens use ID `0`, and unknown tokens use ID `1`.

## Results

The model was evaluated on the 25,000-review test set.

| Metric | Result |
|---|---:|
| Test Accuracy | 81.75% |
| Test Loss | 0.4455 |
| Negative Precision | 77.44% |
| Negative Recall | 89.61% |
| Negative F1-score | 83.08% |
| Positive Precision | 87.67% |
| Positive Recall | 73.89% |
| Positive F1-score | 80.19% |

### Confusion Matrix

Rows represent actual labels, and columns represent predicted labels. The label order is Negative, Positive.

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 11,201 | 1,299 |
| Positive | 3,264 | 9,236 |

The model correctly classified 81.75% of the test reviews. Its negative-class recall was higher than its positive-class recall, indicating that it identified negative reviews more consistently in this evaluation.

These results describe the current model run and may vary if the model is retrained with different settings.

## Project Structure

```text
sentiment-analysis-deep-learning/
├── models/
│   ├── sentiment_lstm.pth
│   └── vocab.json
├── notebooks/
│   └── 01_imdb_sentiment_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── model.py
│   └── predict.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the Repository

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your repository's URL.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd sentiment-analysis-deep-learning
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation requirements depend on your system. For GPU acceleration, install a compatible PyTorch build and CUDA-enabled dependencies. The project can also run on a CPU, although inference and training may be slower.

## Running Predictions

From the project root, run:

```bash
python -m src.predict
```

Enter a movie review when prompted.

Example:

```text
Enter a movie review: This movie was absolutely fantastic! I loved every minute of it.

Sentiment: Positive
Softmax score: 97.34%
Device: cuda
```

The displayed score and device may vary depending on the input and environment.

**Note:** The Softmax score is the model's score for its predicted class. It should not be interpreted as a calibrated probability or a guarantee that the prediction is correct.

## Model Files

The `models/` directory contains:

- `sentiment_lstm.pth` — the trained model weights.
- `vocab.json` — the vocabulary mapping tokens to numerical IDs.

Both files are required to run inference without retraining the model.

## Training and Evaluation

The notebook `notebooks/01_imdb_sentiment_analysis.ipynb` documents the workflow, including:

- Dataset exploration
- Text preprocessing and tokenization
- Vocabulary creation
- Sequence padding and truncation
- PyTorch datasets and data loaders
- Embedding and LSTM model development
- Training and evaluation
- Classification metrics and sample predictions

Open the notebook in Jupyter and execute its cells in order to follow the workflow.

**Important:** Training the notebook again and running its model-saving cell will overwrite the saved model and vocabulary files.

## Future Improvements

Potential improvements include:

- Adding a validation set and early stopping
- Experimenting with embedding dimensions and LSTM hidden sizes
- Evaluating different sequence lengths
- Using pretrained word embeddings
- Comparing LSTM performance with GRU and Transformer-based models
- Improving generalization and probability calibration
- Adding automated unit tests

## License

No license has been specified yet. If you plan to distribute this project for others to use, modify, or redistribute, consider adding an appropriate open-source license.