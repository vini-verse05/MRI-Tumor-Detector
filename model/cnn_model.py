# cnn_model.py
# Purpose: Define CNN architecture using ResNet50 Transfer Learning

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 224


def build_model():
    """
    Build a binary classification CNN using ResNet50 as base.

    Architecture:
      Input (224x224x3)
        ↓
      ResNet50 (pretrained, frozen)  ← Feature extractor
        ↓
      GlobalAveragePooling2D
        ↓
      Dense(256, ReLU) + Dropout(0.5)
        ↓
      Dense(128, ReLU) + Dropout(0.3)
        ↓
      Dense(1, Sigmoid)  → 0=Diseased, 1=Healthy
    """

    # Step 1: Load ResNet50 without its top classification layer
    base_model = ResNet50(
        weights     = 'imagenet',
        include_top = False,
        input_shape = (IMG_SIZE, IMG_SIZE, 3)
    )

    # Step 2: Freeze base model (keep ImageNet knowledge intact)
    base_model.trainable = False

    # Step 3: Build full model on top of ResNet50
    model = models.Sequential([
        base_model,

        # Convert 7x7x2048 feature maps to 2048-dim vector
        layers.GlobalAveragePooling2D(),

        # First fully connected layer
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # Second fully connected layer
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        # Output: 1 neuron, Sigmoid activation
        # Output < 0.5  → Diseased (0)
        # Output >= 0.5 → Healthy  (1)
        layers.Dense(1, activation='sigmoid')
    ])

    # Step 4: Compile
    import tensorflow as tf

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc')   # 🔥 ADD THIS
        ]
    )

    return model, base_model


def unfreeze_top_layers(model, base_model, num_layers=20):
    """
    Fine-tuning: unfreeze the last N layers of ResNet50.
    Call this AFTER initial training.
    """
    base_model.trainable = True

    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    print(f'Unfroze last {num_layers} layers of ResNet50 for fine-tuning')
    return model


# ── Quick Test ──────────────────────────────────────────────────
if __name__ == '__main__':
    model, base = build_model()
    model.summary()