# IMDb Sentiment Analysis with PyTorch

A deep learning project that classifies IMDb movie reviews as **positive** or **negative** using an LSTM (Long Short-Term Memory) neural network implemented with PyTorch.

## Overview

This project uses a recurrent neural network to perform sentiment analysis on movie reviews from the IMDb dataset.

The model processes reviews by converting words into token IDs, mapping them to dense embeddings, and passing the resulting sequences through an LSTM network to predict sentiment.

The model supports two classes:

- **Negative (0)**
- **Positive (1)**

After training, the model weights and vocabulary are saved so that predictions can be made without retraining the network.

## Features

- IMDb movie review sentiment classification
- Text preprocessing and tokenization
- Vocabulary-based numerical encoding
- Embedding and LSTM neural network architecture
- GPU acceleration with CUDA when available
- Saved model weights and vocabulary
- Interactive sentiment prediction for user-provided reviews

## Technologies

- Python
- PyTorch
- NumPy
- Pandas
- NLTK
- Hugging Face Datasets
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Structure

```text
deep-learning/
├── models/
│   ├── sentiment_lstm.pth
│   └── vocab.json
├── notebooks/
│   └── 01_imdb_sentiment_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── model.py
│   ├── predict.py
│   └── preprocess.py
├── README.md
└── requirements.txt
├── .gitignore
```

## Dataset

This project uses the IMDb movie review dataset, available through the Hugging Face Datasets library.

The dataset contains labeled movie reviews for binary sentiment classification.

| Label | Sentiment |
|---|---|
| 0 | Negative |
| 1 | Positive |

The standard dataset split contains 25,000 training reviews and 25,000 test reviews.

## Model Architecture

The model uses the following components:

1. **Embedding Layer:** Converts token IDs into dense vector representations.
2. **LSTM Layer:** Processes the sequence and captures contextual information.
3. **Linear Layer:** Produces scores for the two sentiment classes.

The saved model uses these configuration values:

| Parameter | Value |
|---|---:|
| Embedding dimension | 100 |
| Hidden size | 64 |
| Maximum sequence length | 300 |
| Number of classes | 2 |
| Vocabulary size | 30,000 |

## Installation

Run these commands from the repository root.

### 1. Install dependencies

```bash
pip install -r deep-learning/requirements.txt
```

### 2. Run sentiment prediction

```bash
python deep-learning/src/predict.py
```

Enter a movie review when prompted.

Example:

```text
This movie was amazing and wonderful. I really enjoyed it.
```

Example output:

```text
Sentiment: Positive
Softmax score: 97.58%
Device: cuda
```

The exact score and device may vary depending on the input and hardware.

## Model Evaluation

In the training experiment, the model achieved the following test results:

| Metric | Result |
|---|---:|
| Test accuracy (training evaluation) | 81.75% |
| Test accuracy (comparison evaluation) | 81.74% |
| Test loss (training evaluation) | 0.4455 |

These results describe the recorded experiment and may vary if the model is retrained with different settings or random seeds.

## Saved Artifacts

- `models/sentiment_lstm.pth`: Trained PyTorch model weights.
- `models/vocab.json`: Vocabulary used to encode review text.

Both files are required to load the trained model and perform predictions.

## Limitations

- The model is trained for binary sentiment classification on IMDb movie reviews.
- Performance may be lower on other text domains, such as product reviews or social media posts.
- The softmax score is a model output and should not automatically be interpreted as a calibrated probability.

## Future Improvements

- Extend the existing comparison by testing additional ML and DL architectures.- Experiment with different embedding dimensions and sequence lengths.
- Add more comprehensive automated tests.
- Evaluate performance across different review lengths.
- Explore pretrained word embeddings and transformer-based models.

## Related Project

The repository also contains a separate Machine Learning implementation. See the `../machine-learning/` directory to explore the TF-IDF and Logistic Regression approach.

