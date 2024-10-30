# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from PIL import Image
# import cv2

# # Load the saved model
# model = tf.keras.models.load_model('gender_classification_model4.h5')

# # Define a function for image preprocessing
# def preprocess_image(image, target_size=(256, 256)):
#     # Convert image to RGB if it's not
#     if image.mode != "RGB":
#         image = image.convert("RGB")
    
#     # Resize the image to target size
#     image = image.resize(target_size)
    
#     # Convert the image to a numpy array and scale pixel values
#     image_array = np.array(image) / 255.0
    
#     # Expand dimensions to make it (1, 256, 256, 3) for the model input
#     image_array = np.expand_dims(image_array, axis=0)
#     return image_array

# # Streamlit UI
# st.title("Gender Classification App")
# st.write("Upload an image, and the model will classify the gender as either Male or Female.")

# # File upload section
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Load the image
#     image = Image.open(uploaded_file)
    
#     # Display the image
#     st.image(image, caption="Uploaded Image", use_column_width=True)
    
#     # Preprocess the image
#     processed_image = preprocess_image(image)
    
#     # Make prediction
#     prediction = model.predict(processed_image)
#     predicted_class = "Male" if prediction[0] > 0.5 else "Female"
    
#     # Display the prediction
#     st.write("Prediction:", predicted_class)




















# # app.py

# import streamlit as st
# import numpy as np
# from PIL import Image, ImageOps, ImageDraw
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import img_to_array, load_img

# # Load the pre-trained model
# model = tf.keras.models.load_model("gender_classification_model4.h5")

# # Set Streamlit app title
# st.title("Gender Prediction App")

# # Function to preprocess the uploaded image
# def preprocess_image(image):
#     # Convert to grayscale, resize, and normalize
#     image = ImageOps.grayscale(image)
#     image = image.resize((128, 128))
#     image_array = img_to_array(image) / 255.0
#     image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
#     return image_array

# # Function to display prediction above the image
# def display_prediction(image, gender):
#     # Create a new image with a label box at the top
#     box_height = 50
#     img_with_box = Image.new("RGB", (image.width, image.height + box_height), "white")
#     img_with_box.paste(image, (0, box_height))

#     # Draw the label box
#     draw = ImageDraw.Draw(img_with_box)
#     label = "Female" if gender == 0 else "Male"
#     draw.text((10, 10), f"Prediction: {label}", fill="black")

#     # Display the image with the prediction label
#     st.image(img_with_box, caption="Gender Prediction Result")

# # Upload image section
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Load and display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess and predict gender
#     image_array = preprocess_image(image)
#     prediction = model.predict(image_array)[0][0]
#     gender = 1 if prediction >= 0.5 else 0  # Threshold at 0.5 for male/female

#     # Display prediction above the image
#     display_prediction(image, gender)











 


import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load the saved model
model = tf.keras.models.load_model('gender_classification_model4.h5')

# Set Streamlit app title
st.title("Gender Prediction App")
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