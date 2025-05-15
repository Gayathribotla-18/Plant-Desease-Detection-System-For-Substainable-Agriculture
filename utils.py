import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 128

def load_model():
    model = tf.keras.models.load_model("model/plant_disease_model.h5")
    class_names = ['Healthy', 'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold']  # Edit as needed
    return model, class_names

def predict_disease(image: Image.Image, model, class_names):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    class_idx = np.argmax(predictions)
    confidence = predictions[class_idx] * 100
    return class_names[class_idx], confidence
