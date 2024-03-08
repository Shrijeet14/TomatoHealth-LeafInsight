import streamlit as st
import numpy as np
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Load the model
model = load_model('model.h5')
st.write("Model Loaded Successfully")

def pred_tomato_disease(tomato_plant):
    test_image = load_img(tomato_plant, target_size=(128, 128))
  
    test_image = img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)
  
    result = model.predict(test_image)
  
    pred = np.argmax(result, axis=1)[0]
    return pred

# Streamlit App
st.title("Tomato Disease Prediction")
st.write("Upload an image of a tomato leaf to predict the disease.")

uploaded_file = st.file_uploader("Choose a tomato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file locally
    img = Image.open(uploaded_file)
    file_path = os.path.join("static/upload", uploaded_file.name)
    img.save(file_path)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write('Image uploaded successfully!')

    st.write("RESULT : ")

    # Get the prediction
    pred = pred_tomato_disease(file_path)

    # Map the prediction to disease classes
    classes = [
        "Tomato - Bacteria Spot Disease",
        "Tomato - Early Blight Disease",
        "Tomato - Healthy and Fresh",
        "Tomato - Late Blight Disease",
        "Tomato - Leaf Mold Disease",
        "Tomato - Septoria Leaf Spot Disease",
        "Tomato - Target Spot Disease",
        "Tomato - Tomato Yellow Leaf Curl Virus Disease",
        "Tomato - Tomato Mosaic Virus Disease",
        "Tomato - Two Spotted Spider Mite Disease"
    ]
    print(pred)
    st.write(f"The predicted disease is: {classes[pred]}")
