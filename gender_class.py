import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Gender Prediction",
    page_icon="👫",  # Use the emoji directly
)

# Load the saved model
model = tf.keras.models.load_model('gender_classification_model4.h5')

# Set Streamlit app title
st.title("Gender Prediction App by CNN")
st.write("Upload an image, and the model will classify the gender as either Male or Female.")

# Function to preprocess the uploaded image
def preprocess_image(image, target_size=(128, 128)):
    # Convert image to grayscale, resize, and normalize
    image = ImageOps.grayscale(image)  # Convert to grayscale
    image = image.resize(target_size)  # Resize to target size (128x128)
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# File upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict gender
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]
    gender = "Male" if prediction >= 0.5 else "Female"  # Threshold at 0.5 for male/female

    # Display prediction
    st.markdown(f"<h3 style='text-align: center; color: black;'>Prediction: {gender}</h3>", unsafe_allow_html=True)
