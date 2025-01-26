from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import json

app = Flask(__name__)

# Path for dataset and model
DATASET_PATH = 'C:/Users/Administrator/Desktop/Signscribe/Signscribe/asl_alphabet'
MODEL_PATH = 'sign_language_model.h5'

# Load or create the model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(26, activation='softmax')  # 26 letters (A-Z)
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load dataset and prepare it
def load_and_preprocess_data(dataset_path):
    # Read all images and labels
    images = []
    labels = []
    label_dict = {}  # mapping of class id to letter
    
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            label_dict[len(label_dict)] = folder_name
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (64, 64))  # Resize image
                images.append(image)
                labels.append(len(label_dict) - 1)  # The folder index is the label

    # Convert to numpy arrays
    images = np.array(images).reshape(-1, 64, 64, 1) / 255.0  # Normalize and reshape
    labels = np.array(labels)

    # One-hot encode labels
    labels = to_categorical(labels, num_classes=len(label_dict))

    return images, labels, label_dict

# Save model to a file
def save_model(model, model_path):
    model.save(model_path)

# Train the model
@app.route('/train', methods=['POST'])
def train():
    images, labels, label_dict = load_and_preprocess_data(DATASET_PATH)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = create_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the trained model
    save_model(model, MODEL_PATH)
    
    return jsonify({"message": "Model trained and saved successfully!"}), 200

# Predict using the trained model
@app.route('/predict', methods=['POST'])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not found. Please train the model first."}), 400

    # Load model
    model = load_model(MODEL_PATH)

    # Get the image file from the request
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image uploaded!"}), 400

    # Convert the image to numpy array and preprocess it
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))  # Resize to match the input shape
    img = img.reshape(1, 64, 64, 1) / 255.0  # Normalize

    # Make prediction
    prediction = model.predict(img)
    predicted_label_index = np.argmax(prediction)
    
    # Load label dictionary
    with open('label_dict.json', 'r') as f:
        label_dict = json.load(f)

    predicted_label = label_dict[str(predicted_label_index)]
    return jsonify({"predicted_gesture": predicted_label}), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
