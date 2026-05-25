# evaluate.py
# Computes evaluation metrics for a trained model.

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import pandas as pd


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Runs the model on test data and computes all metrics.

    predict() returns hard labels: 0 or 1
    predict_proba()[:,1] returns fraud probability: 0.0 to 1.0
    We need probabilities for ROC-AUC score.

    Metrics explained:
    - Accuracy: % of all predictions that were correct
    - Precision: of predicted fraud, % that were actually fraud
    - Recall: of actual fraud, % that we caught
    - F1: harmonic mean of precision and recall and balances them into one score
    - ROC-AUC: area under the ROC curve — 0.5=random, 1.0=perfect
    - Confusion matrix: [[TN, FP], [FN, TP]]
      TN=correct legit, FP=false alarm, FN=missed fraud, TP=caught fraud
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc': round(roc_auc_score(y_test, y_proba), 4),
    }

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"  TN={cm[0][0]} FP={cm[0][1]}")
    print(f"  FN={cm[1][0]} TP={cm[1][1]}")

    return metrics


def print_metrics(model_name: str, metrics: dict):
    print(f"\n  {'='*40}")
    print(f"  {model_name} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}  ← most important for fraud")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  {'='*40}")