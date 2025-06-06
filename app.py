from flask import Flask, request, jsonify
import gdown
import tensorflow as tf
import numpy as np
from PIL import Image
from waitress import serve
import cv2
import os

from tensorflow.python.framework import config as tf_config

# Check available devices
print("Devices available to TensorFlow:")
for device in tf.config.list_physical_devices():
    print(device)

# Check the CPU instructions enabled (e.g., AVX2, AVX512)
print("CPU features available:", tf_config.list_logical_devices())

app = Flask(__name__)

IMG_SIZE = (256, 256)

# Model download settings
MODEL_ID = '1BzSiFrNv5QI_-Ehea22gLHOlSzxATUzO'  # Replace with your actual Google Drive file ID
MODEL_FILE = 'Test10.keras'
MODEL_URL = f'https://drive.google.com/uc?id={MODEL_ID}'
MODEL_DIR = os.path.join(os.getcwd(), 'model')  # Use current working directory to construct the path

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Function to download the model if it doesn't exist
def download_model():
    # Full path to the model file
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    
    # Debugging: Print current working directory
    print("Current working directory:", os.getcwd())
    print(f"Checking model at path: {model_path}")
    
    # Check if the model file exists locally
    if not os.path.exists(model_path):
        print(f'Model not found locally, downloading from Google Drive...')
        
        try:
            # Download the model from Google Drive
            gdown.download(MODEL_URL, model_path, quiet=False)
            print(f"Model downloaded successfully to {model_path}")
        except Exception as e:
            print(f"Error downloading the model: {e}")
            raise
    
    # Try to load the model from the path
    try:
        print(f"Loading model from {model_path}")
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading the model: {e}")
        raise

# Load the model on server start
model = download_model()

def actualise_color(image):
    if not isinstance(image, np.ndarray):
        image_np = image.numpy()
    image_np = image_np.astype(np.uint8)
    image_np = np.clip(image_np / 255, 0, 1).astype(np.float32)
    return image_np

def apply_clahe_np(image):
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3D RGB image, got shape: {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return np.clip(final / 255, 0, 1).astype(np.float32)

def convert_to_hsv(image):
    image = tf.image.convert_image_dtype(image, tf.float32)  # Ensure in [0, 1]
    image_hsv = tf.image.rgb_to_hsv(image)
    return image_hsv

def color_based_subtraction_cv(image):
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    h, s, v = np.split(image, 3, axis=-1)
    green_mask = (h > 0.215) & (h < 0.55) & (s > 0.21) & (v > 0.5)
    mask = np.concatenate([green_mask] * 3, axis=-1).astype(np.float32)
    result = image * mask
    return result

def opencv_morph(image):
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray, 13, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_DILATE, kernel)
    opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, kernel)
    opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    mask_rgb = cv2.cvtColor(opened, cv2.COLOR_GRAY2RGB)
    result = cv2.bitwise_and(image, mask_rgb)
    return result.astype(np.float32) / 255.0

def preprocess_image(image_file):
    # Open the image using PIL
    image = Image.open(image_file)
    # Resize image to the target size
    image = image.resize(IMG_SIZE)
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Convert to TensorFlow tensor
    tensor_image = tf.convert_to_tensor(image_array, dtype=tf.float32)

    # Preprocessing pipeline
    tensor_image = actualise_color(tensor_image)
    tensor_image = apply_clahe_np(tensor_image)
    tensor_image = convert_to_hsv(tensor_image)
    tensor_image = color_based_subtraction_cv(tensor_image)
    tensor_image = opencv_morph(tensor_image)

    # Expand the dimension to match model input (batch size of 1)
    tensor_image = np.expand_dims(tensor_image, axis=0)
    return tensor_image

@app.route('/', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    

    # Preprocess the image
    tensor_image = preprocess_image(file)
    
    # Make a prediction
    predictions = model.predict(tensor_image)
    predicted_class_index = np.argmax(predictions[0])
    confidence_score = predictions[0][predicted_class_index]
    
    return jsonify({
        'predicted_class_index': int(predicted_class_index),
        'confidence_score': float(confidence_score)
    })

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))