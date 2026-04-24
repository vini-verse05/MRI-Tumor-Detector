# train_model.py
# Purpose: Train the CNN model and save it to disk

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.preprocessing import get_data_generators
from model.cnn_model import build_model, unfreeze_top_layers

os.makedirs('model', exist_ok=True)

# ── Step 1: Load data ──────────────────────────────────────────
print('Loading dataset...')
train_gen, val_gen, test_gen = get_data_generators()

print("Class indices:", train_gen.class_indices)

# ── Step 2: Compute class weights to fix class imbalance ───────
labels = train_gen.classes
class_weights_arr = compute_class_weight(
    class_weight = 'balanced',
    classes      = np.unique(labels),
    y            = labels
)
class_weight_dict = dict(enumerate(class_weights_arr))
print(f'Class weights: {class_weight_dict}')
# Example: {0: 0.62, 1: 2.8}  (Healthy gets higher weight)

# ── Step 3: Build model ────────────────────────────────────────
print('Building model...')
model, base_model = build_model()

# ── Step 4: Callbacks ──────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        filepath       = 'model/brain_tumor_model.h5',
        monitor        = 'val_accuracy',
        save_best_only = True,
        verbose        = 1
    ),
    EarlyStopping(
        monitor              = 'val_accuracy',
        patience             = 5,
        restore_best_weights = True,
        verbose              = 1
    ),
    ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.5,
        patience = 3,
        verbose  = 1
    )
]

# ── Step 5: Phase 1 — Train with frozen ResNet50 ───────────────
print('\nPhase 1: Training custom layers only (ResNet50 frozen)...')
history1 = model.fit(
    train_gen,
    epochs          = 20,
    validation_data = val_gen,
    class_weight    = class_weight_dict,
    callbacks       = callbacks,
    verbose         = 1
)

# ── Step 6: Phase 2 — Fine-tune top ResNet50 layers ───────────
print('\nPhase 2: Fine-tuning top ResNet50 layers...')
model = unfreeze_top_layers(model, base_model, num_layers=20)

history2 = model.fit(
    train_gen,
    epochs          = 15,
    validation_data = val_gen,
    class_weight    = class_weight_dict,
    callbacks       = callbacks,
    verbose         = 1
)

# ── Step 7: Evaluate on test set ──────────────────────────────
print('\nEvaluating on test set...')
loss, accuracy, auc = model.evaluate(test_gen)

print(f'Test Accuracy: {accuracy * 100:.2f}%')
print(f'Test Loss    : {loss:.4f}')
print(f'Test AUC     : {auc:.4f}')

# ── Step 8: Plot training curves ──────────────────────────────
def plot_history(h1, h2):
    acc      = h1.history['accuracy']     + h2.history['accuracy']
    val_acc  = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss     = h1.history['loss']         + h2.history['loss']
    val_loss = h1.history['val_loss']     + h2.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc,     'b-', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.xlabel('Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss,     'b-', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'r-', label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.xlabel('Epoch')

    plt.tight_layout()
    plt.savefig('model/training_curves.png', dpi=150)
    print('Training curves saved to models/training_curves.png')

plot_history(history1, history2)
print('\nTraining complete! Model saved to model/brain_tumor_model.h5')