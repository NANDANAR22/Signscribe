# frontend.py
import streamlit as st
from PIL import Image

def main():
    st.title("Signscribe - Sign Language Recognition")

    # Image upload component
    uploaded_image = st.file_uploader("Upload an image for sign language recognition", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Processing the image...")  # Placeholder for backend logic

if __name__ == "__main__":
    main()
