import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import geocoder
import random

# Load the TensorFlow Lite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path='model_unquant.tflite')
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Get input and output details for the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the input size of your model
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to decode the prediction to a tomato disease label
def decode_prediction(prediction):
    class_labels = [
        'Powdery_Mildew', 'Healthy', 'Tomato_mosaic_virus',
        'Tomato_yellow_leaf_curl_virus', 'Tomato_Target_Spot',
        'Tomato_Spider_Mites', 'Septoria_leaf_Spot', 'Leaf_Mold',
        'Tomato_late_Blight', 'Tomato_Early_Blight', 'Tomato_bacterial_Spot'
    ]
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_labels[predicted_class] if predicted_class < len(class_labels) else "Unknown Class"

# Disease causes and treatments
disease_info = {
    'Powdery_Mildew': {'causes': "Fungal disease caused by Erysiphales.", 'treatment': "Apply fungicides and ensure proper ventilation."},
    'Healthy': {'causes': "No disease detected, the plant appears healthy.", 'treatment': "No treatment necessary. Continue regular care."},
    'Tomato_mosaic_virus': {'causes': "Caused by the Tomato Mosaic Virus.", 'treatment': "Remove infected plants and use virus-free seeds."},
    'Tomato_yellow_leaf_curl_virus': {'causes': "Spread by whiteflies.", 'treatment': "Control whiteflies and remove infected plants."},
    'Tomato_Target_Spot': {'causes': "Fungal disease caused by Corynespora cassiicola.", 'treatment': "Use fungicides and improve air circulation."},
    'Tomato_Spider_Mites': {'causes': "Caused by Tetranychidae mites.", 'treatment': "Use miticides and maintain proper humidity levels."},
    'Septoria_leaf_Spot': {'causes': "Caused by the fungus Septoria lycopersici.", 'treatment': "Apply fungicides and remove affected leaves."},
    'Leaf_Mold': {'causes': "Caused by the fungus Passalora fulva.", 'treatment': "Ensure ventilation and use fungicides."},
    'Tomato_late_Blight': {'causes': "Caused by Phytophthora infestans.", 'treatment': "Remove infected plants and use fungicides."},
    'Tomato_Early_Blight': {'causes': "Caused by Alternaria solani.", 'treatment': "Apply fungicides and rotate crops."},
    'Tomato_bacterial_Spot': {'causes': "Caused by Xanthomonas species.", 'treatment': "Use copper-based bactericides and avoid overhead watering."}
}

# Function to predict diseases based on conditions
def predict_disease_based_on_conditions(temp, humidity, pH, npk, moisture):
    diseases = []
    if pH < 5.5 and moisture > 60:
        diseases.append("Tomato_late_Blight")
    if humidity > 60 and temp > 15 and temp < 30:
        diseases.append("Leaf_Mold")
    if humidity > 70 and temp < 25:
        diseases.append("Tomato_Target_Spot")
    return diseases if diseases else ["No specific disease identified based on conditions."]

# Automatically detect user's location
def get_coordinates():
    g = geocoder.ip('me')
    return g.latlng if g.ok else (None, None)

# Streamlit UI
st.title("Tomato Disease Detection and Prediction Using ML")

# Get coordinates
coordinates = get_coordinates()
if coordinates:
    latitude, longitude = coordinates
    st.write(f"Detected Coordinates: Latitude = {latitude}, Longitude = {longitude}")
else:
    st.write("Unable to detect coordinates. Please enter manually.")

# Upload image
uploaded_file = st.file_uploader("Upload an image of the affected area...", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess and run inference
    processed_image = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    disease = decode_prediction(output_data)
    
    st.write(f"Disease Detected: **{disease}**")
    disease_details = disease_info.get(disease, {"causes": "No information available.", "treatment": ""})
    st.write(f"**Causes:** {disease_details['causes']}")
    st.write(f"**Treatment:** {disease_details['treatment']}")

    if coordinates:
        temperature, humidity = 25, 65  # Example data
        st.write(f"Temperature: {temperature}°C, Humidity: {humidity}%")
        pH = round(random.uniform(4.5, 7.5), 2)
        npk = (round(random.uniform(10, 20), 1), round(random.uniform(5, 15), 1), round(random.uniform(5, 15), 1))
        moisture = round(random.uniform(40, 80), 1)
        st.write(f"Soil pH: {pH}, Soil NPK: {npk}, Moisture: {moisture}%")
        possible_diseases = predict_disease_based_on_conditions(temperature, humidity, pH, npk, moisture)
        st.write(f"**Possible Diseases Based on Conditions:** {', '.join(possible_diseases)}")
        for disease in possible_diseases:
            if disease in disease_info:
                st.write(f"**{disease} Treatment:** {disease_info[disease]['treatment']}")
                st.write(f"**{disease} Causes:** {disease_info[disease]['causes']}")
