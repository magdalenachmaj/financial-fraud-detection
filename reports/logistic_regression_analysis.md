# Logistic Regression – Model Evaluation Report

## 1. Objective

The objective of the initial experiment was to establish a baseline machine learning model for credit card fraud detection.

Logistic Regression was selected as the first model due to its simplicity, interpretability, and suitability as a classification baseline.

## 2. Class Imbalance

The dataset contains 284,807 transactions, of which only 492 are fraudulent.

Fraudulent transactions represent approximately 0.173% of the dataset.

This imbalance makes accuracy an unreliable standalone performance measure.

To address this issue, the model was trained using:

- `class_weight="balanced"`
- Stratified train-test splitting
- Feature scaling within a Scikit-learn Pipeline

## 3. Baseline Test Evaluation

The initial model was evaluated using a classification threshold of 0.50.

| Metric | Result |
|---|---:|
| Fraud Precision | 0.06 |
| Fraud Recall | 0.91 |
| Fraud F1-score | 0.11 |
| PR-AUC | 0.7241 |
| ROC-AUC | 0.9733 |

### Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 55,375 | 1,489 |
| Actual Fraud | 9 | 89 |

### Interpretation

The model successfully detected 89 of the 98 fraudulent transactions.

However, it incorrectly classified 1,489 legitimate transactions as fraudulent.

The high Recall indicates that the model is effective at identifying fraudulent transactions, but the low Precision demonstrates that the default threshold produces too many false positives.

The ROC-AUC of 0.9733 indicates strong ranking performance. Nevertheless, the Precision-Recall trade-off is more informative for this highly imbalanced classification problem.

## 4. Validation-Based Threshold Optimization

A separate validation subset was used to evaluate different classification thresholds.

The threshold was not selected using the test set.

| Threshold | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.50 | 0.058 | 0.886 | 0.108 |
| 0.70 | 0.104 | 0.848 | 0.185 |
| 0.80 | 0.140 | 0.835 | 0.240 |
| 0.90 | 0.193 | 0.835 | 0.314 |
| 0.95 | 0.258 | 0.810 | 0.391 |
| 0.98 | 0.368 | 0.810 | 0.506 |
| 0.99 | 0.481 | 0.797 | 0.600 |

### Selected Validation Threshold

The threshold of 0.99 achieved the highest F1-score among the evaluated thresholds.

Validation confusion matrix counts:

| Metric | Count |
|---|---:|
| True Positives | 63 |
| False Positives | 68 |
| False Negatives | 16 |
| True Negatives | 45,422 |

At this threshold:

- Precision increased to 48.1%.
- Recall remained relatively high at 79.7%.
- F1-score increased to 0.600.

The results demonstrate that threshold selection has a substantial impact on fraud detection performance.

## 5. Key Findings

1. The default threshold prioritizes fraud detection but generates a large number of false positives.
2. Increasing the classification threshold improves Precision.
3. The validation threshold of 0.99 provides a better Precision-Recall balance according to F1-score.
4. ROC-AUC alone is insufficient to determine the operational usefulness of a fraud detection model.
5. The threshold should ultimately reflect the relative costs of missed fraud and unnecessary transaction reviews.

## 6. Limitations

- The dataset is highly imbalanced.
- The anonymized PCA features limit direct business interpretation.
- The random stratified split does not fully simulate future transaction data.
- The threshold of 0.99 has only been selected on validation data.
- Final test-set performance at the selected threshold remains to be evaluated.

## 7. Conclusion

Logistic Regression provides a useful baseline for the fraud detection task.

The initial results demonstrate strong ranking performance and high fraud Recall at the default threshold, but also reveal a substantial false positive problem.

Validation-based threshold optimization significantly improves the Precision-F1 balance.

The next experiment will investigate whether tree-based ensemble methods, particularly Random Forest, can improve detection performance.

**Evaluation status:** Initial baseline test evaluation completed. Final test evaluation using the selected validation threshold is pending.