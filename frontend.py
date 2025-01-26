import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model("sign_language_model.h5")

# Load the label dictionary
@st.cache_data
def load_label_dict():
    # Replace with the actual label dictionary used during training
    return {
        0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
        5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
        10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
        15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
        20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
    }

# Preprocess the uploaded image for prediction
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (64, 64))  # Resize to match the input shape
    image = image.reshape(1, 64, 64, 1) / 255.0  # Normalize
    return image

# Perform prediction
def predict(image, model, label_dict):
    prediction = model.predict(image)
    gesture_index = np.argmax(prediction)
    gesture = label_dict[gesture_index]
    return gesture

# Streamlit UI
def main():
    st.title("Sign Language Recognition")
    st.write("Upload an image of a sign language gesture to get a prediction.")

    # File uploader for image input
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Load model and labels
        model = load_trained_model()
        label_dict = load_label_dict()

        # Predict the gesture
        prediction = predict(processed_image, model, label_dict)

        st.write("## Predicted Gesture:")
        st.success(prediction)

# Corrected if condition to check if the script is executed directly
if __name__ == "__main__":
    main()
