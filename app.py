import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import google.generativeai as genai

app = Flask(__name__)

# --- CONFIGURATION ---
# REPLACE THIS with your actual Google Gemini API Key
GOOGLE_API_KEY = "AIzaSyCo3t5qNeYL7vgElt_I_G5-sr4Ew-Epmww"
genai.configure(api_key=GOOGLE_API_KEY)

# Load Model and Class Names
model = load_model('model.h5')

with open('classes.json', 'r') as f:
    class_names = json.load(f)

# --- HELPER FUNCTIONS ---
def get_gemini_cure(disease_name):
    try:
        model_gemini = genai.GenerativeModel('gemini-pro')
        prompt = f"Provide a concise treatment and prevention plan for the plant disease: {disease_name}. Keep it under 100 words."
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return "Cure information currently unavailable from AI."

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Save temp file
    file_path = os.path.join('static', file.filename)
    os.makedirs('static', exist_ok=True)
    file.save(file_path)

    # Prediction
    processed_img = prepare_image(file_path)
    prediction = model.predict(processed_img)
    
    # Logic for Unknown Images (Thresholding)
    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction)
    
    # Threshold: If confidence is less than 70%, consider it unknown
    if confidence < 0.70:
        os.remove(file_path) # Clean up
        return jsonify({
            'disease': "Unknown",
            'cure': "Sorry, we are learning. Please try another image."
        })

    disease_name = class_names[str(predicted_class)]
    
    # Get Cure from Gemini
    cure_info = get_gemini_cure(disease_name)
    
    # Clean up temp file
    os.remove(file_path)

    return jsonify({
        'disease': disease_name.replace("_", " "),
        'cure': cure_info
    })

if __name__ == '__main__':
    app.run(debug=True)