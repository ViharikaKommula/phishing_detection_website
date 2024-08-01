from flask import Flask, request, jsonify, render_template, session
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load pre-trained model and scaler
model = load_model('phishing_detection_model.h5')
scaler = joblib.load('scaler.pkl')

def extract_features(url):
    # Feature extraction must match the training script
    return np.array([len(url), url.count('.'), url.count('/'), url.count('-')])

def preprocess_input(url):
    features = extract_features(url)
    standardized_features = scaler.transform([features])
    return standardized_features[0]

@app.route('/')
def home():
    history = session.get('history', [])
    return render_template('index.html', history=history)

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    processed_url = preprocess_input(url)
    prediction = model.predict(np.array([processed_url]))
    confidence = float(prediction[0][0])
    result = 'Phishing' if confidence > 0.5 else 'Legitimate'
    
    # Update session history
    if 'history' not in session:
        session['history'] = []
    session['history'].append({'url': url, 'result': result, 'confidence': confidence})

    return jsonify({'prediction': result, 'confidence': confidence})

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback = request.form['feedback']
    url = request.form['url']
    # Here you would typically store feedback in a database
    return jsonify({'status': 'Feedback received', 'url': url, 'feedback': feedback})

if __name__ == "__main__":
    app.run(debug=True)
