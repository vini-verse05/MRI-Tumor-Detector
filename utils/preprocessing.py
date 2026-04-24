# preprocessing.py
# Purpose: Load, resize, normalize, and augment MRI images

import os
import cv2
import numpy as np

# ── Constants ──────────────────────────────────────────────────
IMG_SIZE   = 224        # ResNet50 expects 224x224 pixels
BATCH_SIZE = 32
TRAIN_DIR  = 'dataset/train'
TEST_DIR   = 'dataset/test'


def preprocess_single_image(image_path):
    """
    Preprocess a single MRI image for prediction.
    Returns a numpy array of shape (1, 224, 224, 3).
    """
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f'Could not read image: {image_path}')

    # Convert BGR → RGB (ResNet50 was trained on RGB images)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to 224x224
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize pixel values to [0, 1]
    img = img.astype('float32') / 255.0

    # Add batch dimension: (224,224,3) → (1,224,224,3)
    img = np.expand_dims(img, axis=0)

    return img


def get_data_generators():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    """
    Create training and validation data generators with augmentation.
    """
    # Training generator WITH augmentation
    train_datagen = ImageDataGenerator(
        rescale            = 1.0/255,
        rotation_range     = 15,
        width_shift_range  = 0.1,
        height_shift_range = 0.1,
        shear_range        = 0.1,
        zoom_range         = 0.1,
        horizontal_flip    = True,
        validation_split   = 0.2    # 20% of train → validation
    )

    # Test generator WITHOUT augmentation
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    # Training subset (80% of train folder)
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size  = BATCH_SIZE,
        class_mode  = 'binary',    # 0=Diseased, 1=Healthy
        subset      = 'training',
        shuffle     = True,
        seed        = 42
    )

    # Validation subset (20% of train folder)
    val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size  = BATCH_SIZE,
        class_mode  = 'binary',
        subset      = 'validation',
        shuffle     = False,
        seed        = 42
    )

    # Test generator
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size  = BATCH_SIZE,
        class_mode  = 'binary',
        shuffle     = False
    )

    return train_generator, val_generator, test_generator


# ── Quick Test ──────────────────────────────────────────────────
if __name__ == '__main__':
    train_gen, val_gen, test_gen = get_data_generators()
    print(f'Training samples  : {train_gen.samples}')
    print(f'Validation samples: {val_gen.samples}')
    print(f'Test samples      : {test_gen.samples}')
    print(f'Class mapping     : {train_gen.class_indices}')
    # Expected: {'Diseased': 0, 'Healthy': 1}