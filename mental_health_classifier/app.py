from flask import Flask, render_template, request, jsonify, make_response, send_from_directory
import pickle
import pandas as pd
from textblob import TextBlob
import traceback
import logging
from datetime import datetime
import os

app = Flask(__name__, static_url_path='/static')
logging.basicConfig(level=logging.DEBUG)

# Load the model and components
try:
    model = pickle.load(open('mental_health_classification_nn_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    le = pickle.load(open('label_encoder.pkl', 'rb'))
except Exception as e:
    app.logger.error(f"Error loading model components: {str(e)}")

# Feature extraction function
def extract_features(df):
    df['text_length'] = df['statement'].apply(len)
    df['polarity'] = df['statement'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['statement'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df['word_count'] = df['statement'].apply(lambda x: len(x.split()))
    df['avg_word_length'] = df['statement'].apply(
        lambda x: sum(len(word) for word in x.split()) / len(x.split()) if len(x.split()) > 0 else 0
    )
    return df

numerical_features = ['text_length', 'polarity', 'subjectivity', 'word_count', 'avg_word_length']

# New function to log user input and prediction
def log_prediction(user_input, prediction, probabilities):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} | Input: {user_input} | Prediction: {prediction} | Probabilities: {probabilities}\n"
    log_file_path = 'user_predictions.log'
    with open(log_file_path, 'a') as log_file:
        log_file.write(log_entry)
    app.logger.info(f"Logged prediction: {log_entry}")

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/')
def home():
    return make_response(render_template('index.html'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form['text']
        app.logger.debug(f"Received text: {text}")

        # Preprocess the new input
        new_input = pd.DataFrame({'statement': [text]})
        new_input = extract_features(new_input)

        new_input_vectorized = vectorizer.transform([text])
        new_input_features = pd.DataFrame(new_input_vectorized.toarray(), columns=vectorizer.get_feature_names_out())

        # Add numerical features
        for feature in numerical_features:
            new_input_features[feature] = new_input[feature]

        # Scale numerical features
        new_input_features[numerical_features] = scaler.transform(new_input_features[numerical_features])

        # Make prediction
        prediction = model.predict(new_input_features)
        prediction_proba = model.predict_proba(new_input_features)

        predicted_class = le.inverse_transform(prediction)[0]
        class_probabilities = dict(zip(le.classes_, prediction_proba[0]))

        # Log the prediction
        log_prediction(text, predicted_class, class_probabilities)

        app.logger.debug(f"Prediction: {predicted_class}")
        app.logger.debug(f"Probabilities: {class_probabilities}")

        return jsonify({
            'prediction': predicted_class,
            'probabilities': class_probabilities
        })
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)