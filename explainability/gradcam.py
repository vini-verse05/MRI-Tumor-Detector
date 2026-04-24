# gradcam.py
# Purpose: Generate Grad-CAM heatmaps for model explainability

import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def get_gradcam_heatmap(model, img_array, last_conv_layer_name=None):

    # Ensure correct shape
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)

    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # ✅ Get base ResNet model
    base_model = model.get_layer('resnet50')

    # ✅ Get last conv layer
    last_conv_layer = base_model.get_layer('conv5_block3_out')

    # ✅ Create a model ONLY up to last conv layer
    conv_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    # ✅ Get classifier (top layers AFTER ResNet)
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input

    # Reuse top layers from original model
    for layer in model.layers[1:]:  # skip resnet50
        x = layer(x)

    classifier_model = tf.keras.models.Model(classifier_input, x)

    # ✅ Compute Grad-CAM
    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)

        predictions = classifier_model(conv_outputs)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

def save_gradcam_image(img_path, heatmap, output_filename, output_dir, alpha=0.4):
    """
    Overlay the Grad-CAM heatmap on the original image and save.
    """

    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (224, 224))

    # Convert to color
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))  # 🔥 MUCH BIGGER

    axes[0].imshow(img)
    axes[0].set_title('Original MRI')
    axes[0].axis('off')

    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)  # 🔥 HIGH QUALITY
    plt.close()

    print(f'Grad-CAM saved to: {output_path}')
    return f"/static/heatmaps/{output_filename}"


def find_last_conv_layer(model):
    """
    Automatically find last conv layer (fallback utility)
    """
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            print(f'Last conv layer: {layer.name}')
            return layer.name
    raise ValueError('No convolutional layer found!')