# Supervised SBERT evaluation

Labels are derived from ratings: 1–2 negative, 3 neutral, 4–5 positive. They are weak labels, not manually verified ground truth.

| Model | Accuracy | Macro F1 | Weighted F1 | Macro ROC-AUC |
|---|---:|---:|---:|---:|
| sbert_reference_baseline | 0.5778 | 0.4090 | 0.6318 | n/a |
| logistic_regression | 0.7045 | 0.5598 | 0.7475 | 0.8504337788881777 |
| linear_svm | 0.8075 | 0.5844 | 0.8066 | n/a |
| mlp | 0.7202 | 0.5777 | 0.7630 | 0.8655757472189802 |
