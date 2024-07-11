import streamlit as st
import tensorflow as tf
from keras.saving import load_model
from PIL import Image, ImageOps
import numpy as np
import requests
from io import BytesIO

# Load your trained model
model = load_model('model1.keras')


# Define a function to preprocess the uploaded image
def preprocess_image(image):
    size = (32, 32)  # Adjust the size as per your model input size
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0,1] range
    return img_array


# Define a function to predict skin cancer
def predict(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction


# Custom CSS for better UI
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 5px;
        border: none;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stFileUploader {
        margin-bottom: 20px;
    }
    .stTextInput>div>input {
        background-color: white;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit interface
st.title("Skin Cancer Prediction")
st.write(
    """
    **Upload an image** of the skin lesion to predict its type, or **provide an image URL**. 
    This tool uses a machine learning model to classify the lesion into one of the following types:
    - Actinic keratoses and intraepithelial carcinoma
    - Basal cell carcinoma
    - Benign lesions of the keratosis
    - Dermatofibroma
    - Melanoma
    - Melanocytic nevi
    - Vascular lesions
    """
)

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# URL input
url = st.text_input("Or enter an image URL:")

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif url:
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        st.error("Error fetching the image from the URL. Please ensure the URL is correct and points to a valid image.")

if image is not None:
    st.image(image, caption='Uploaded Image', width=300)
    st.write("Classifying...")
    prediction = predict(image, model)

    # Assuming your model outputs a probability for each class
    class_names = [
        'Actinic keratoses and intraepithelial carcinoma',
        'Basal cell carcinoma',
        'Benign lesions of the keratosis',
        'Dermatofibroma',
        'Melanoma',
        'Melanocytic nevi',
        'Vascular lesions'
    ]
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100

    st.write(
        f"The lesion is predicted to be: **{class_names[predicted_class]}** with **{confidence:.2f}%** confidence.")

    # Display confidence scores for all classes
    st.write("Confidence scores for other classes:")
    for i, class_name in enumerate(class_names):
        if i != predicted_class:
            st.write(f"- {class_name}: {prediction[0][i] * 100:.2f}%")
