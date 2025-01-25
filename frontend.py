import streamlit as st
from PIL import Image
import numpy as np
import io
import cv2
from backend import process_image  

def main():
    st.title("Signscribe - Sign Language Recognition")

    # Image upload component
    uploaded_image = st.file_uploader("Upload an image for sign language recognition", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Processing the image...")

       
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)  

       
        processed_image, gesture = process_image(image_bgr)

      
        st.write(f"Detected Gesture: {gesture}")

        
        processed_image_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        st.image(processed_image_pil, caption="Processed Image with Hand Landmarks", use_column_width=True)

if _name_ == "_main_":
    main()
