# IMDb Sentiment Analysis: Machine Learning vs Deep Learning

This directory compares two sentiment classification approaches on the IMDb movie review dataset.

## Models

| Model | Approach |
|---|---|
| Machine Learning | TF-IDF + Logistic Regression |
| Deep Learning | Embedding + LSTM (PyTorch) |

Both models were evaluated on the same 25,000 IMDb test reviews, using their respective preprocessing pipelines and saved model artifacts. Neither model was retrained during the comparison.

## Evaluation Results

| Metric | TF-IDF + Logistic Regression | Embedding + LSTM |
|---|---:|---:|
| Accuracy | **88.32%** | 81.74% |
| Precision | 88.11% | **87.67%** |
| Recall | **88.61%** | 73.88% |
| F1-score | **88.36%** | 80.19% |
| Inference time (25,000 reviews) | **4.707 s** | 49.676 s |
| Average latency per review | **0.188 ms** | 1.987 ms |
| Required artifact size | **0.887 MB** | 12.120 MB |

*Inference timings are from a single recorded run and should be treated as indicative rather than definitive benchmarks. They include preprocessing and prediction as implemented in this notebook.*

## Performance Visualization

![Model performance comparison](model_performance.png)

## Confusion Matrices

### TF-IDF + Logistic Regression

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 11,005 | 1,495 |
| Positive | 1,424 | 11,076 |

### Embedding + LSTM

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 11,201 | 1,299 |
| Positive | 3,265 | 9,235 |

## Analysis

- **Overall performance:** TF-IDF + Logistic Regression achieved higher accuracy and F1-score in this experiment.
- **Positive sentiment:** Logistic Regression had substantially higher recall for positive reviews.
- **Negative sentiment:** LSTM correctly identified more negative reviews, but it missed more positive reviews by classifying them as negative.
- **Inference speed:** The recorded Logistic Regression pipeline was faster in this run. Timing comparisons depend on hardware, batch size, and preprocessing implementation.
- **Storage:** The LSTM requires approximately 12.12 MB for its weights and vocabulary, compared with 0.887 MB for the saved ML pipeline.

## Conclusion

For this particular IMDb experiment, TF-IDF + Logistic Regression provided the strongest combination of predictive performance, inference speed, and storage efficiency. The LSTM experiment remains useful for studying neural sequence models and provides a baseline for future improvements.

These results apply to the current model configurations and evaluation procedure; they do not establish that traditional machine learning is universally better than deep learning.

## Reproducibility

The notebook used to generate the results is `model_comparison.ipynb`.

The comparison uses the saved model artifacts and the original IMDb test split. Each model uses its own preprocessing pipeline.

Training times are not reported because they were not measured and recorded as part of this comparison.