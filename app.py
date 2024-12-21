import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
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

# Simulate weather data for Bangalore
def get_simulated_weather():
    # Typical weather ranges in Bangalore
    temperature = round(random.uniform(20, 35), 1)  # Temperature in Celsius
    humidity = round(random.uniform(50, 80), 1)     # Humidity in percentage
    return temperature, humidity

# Simulate soil data based on Bangalore's conditions
def get_simulated_soil_data():
    pH = round(random.uniform(5.5, 7.0), 2)  # Soil pH for Bangalore
    npk = (round(random.uniform(15, 25), 1),  # Nitrogen (N)
           round(random.uniform(10, 20), 1),  # Phosphorus (P)
           round(random.uniform(10, 20), 1))  # Potassium (K)
    moisture = round(random.uniform(50, 70), 1)  # Soil moisture in percentage
    return pH, npk, moisture

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

# Streamlit UI
st.title("Tomato Disease Detection and Prediction Using ML")

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
    
    st.subheader("Detection")
    st.write(f"Disease Detected: **{disease}**")
    disease_details = disease_info.get(disease, {"causes": "No information available.", "treatment": ""})
    st.write(f"**Causes:** {disease_details['causes']}")
    st.write(f"**Treatment:** {disease_details['treatment']}")

    # Simulate and display weather and soil conditions
    temperature, humidity = get_simulated_weather()
    pH, npk, moisture = get_simulated_soil_data()

    st.subheader("Weather Conditions")
    st.write(f"Temperature: {temperature}°C, Humidity: {humidity}%")
    
    st.subheader("Soil Conditions")
    st.write(f"pH: {pH}, NPK: {npk}, Moisture: {moisture}%")

    # Predict diseases based on conditions
    possible_diseases = predict_disease_based_on_conditions(temperature, humidity, pH, npk, moisture)
    
    st.subheader("Prediction")
    st.write(f"**Possible Diseases Based on Conditions:** {', '.join(possible_diseases)}")
    for disease in possible_diseases:
        if disease in disease_info:
            st.write(f"**{disease} Treatment:** {disease_info[disease]['treatment']}")
            st.write(f"**{disease} Causes:** {disease_info[disease]['causes']}")
    
    # Personal suggestions and alternatives
    st.subheader("Suggestions")
    st.write("Consider crop rotation with resistant plants like spinach or lettuce to minimize future risks.")
    st.write("Ensure proper ventilation, avoid overwatering, and use organic fertilizers for soil enrichment.")
