# evaluate_metrics.py
# Run this after training to compute all evaluation metrics

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
import tensorflow as tf
from utils.preprocessing import get_data_generators

# Load model and test data
model    = tf.keras.models.load_model('model/brain_tumor_model.h5')
_, _, test_gen = get_data_generators()

# Get predictions
y_true = test_gen.classes
y_prob = model.predict(test_gen, verbose=1).flatten()
y_pred = (y_prob >= 0.5).astype(int)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f'\nConfusion Matrix:')
print(f'  True Negatives  (TN): {tn}  ← Healthy correctly identified')
print(f'  False Positives (FP): {fp}  ← Healthy wrongly called Diseased')
print(f'  False Negatives (FN): {fn}  ← MISSED TUMORS (minimize this!)')
print(f'  True Positives  (TP): {tp}  ← Diseased correctly identified')

# Metrics
accuracy    = accuracy_score(y_true, y_pred) * 100
sensitivity = tp / (tp + fn) * 100   # Recall
specificity = tn / (tn + fp) * 100
precision   = tp / (tp + fp) * 100
f1          = 2 * (precision * sensitivity) / (precision + sensitivity)
auc         = roc_auc_score(y_true, y_prob)

print(f'\n--- Evaluation Metrics ---')
print(f'Accuracy    : {accuracy:.2f}%')
print(f'Sensitivity : {sensitivity:.2f}%  (most important for medical AI)')
print(f'Specificity : {specificity:.2f}%')
print(f'F1-Score    : {f1:.2f}%')
print(f'AUC-ROC     : {auc:.4f}')
print(f'\nDetailed Report:')
print(classification_report(y_true, y_pred,
      target_names=['Diseased (0)', 'Healthy (1)']))

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_true, y_pred)

# optimal threshold
optimal_idx = (tpr - fpr).argmax()
optimal_threshold = thresholds[optimal_idx]

print("Best Threshold:", optimal_threshold)