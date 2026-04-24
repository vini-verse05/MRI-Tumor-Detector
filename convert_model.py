import tensorflow as tf

model = tf.keras.models.load_model('model/brain_tumor_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantization reduces size further
tflite_model = converter.convert()

with open('model/brain_tumor_model.tflite', 'wb') as f:
    f.write(tflite_model)

print('Converted! Size:', round(len(tflite_model) / 1024 / 1024, 2), 'MB')
