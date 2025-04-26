from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# 🔵 Home page with HTML
@app.route('/')
def home():
    return '''
    <html>
      <head>
        <title>Cybersecurity Prediction API</title>
      </head>
      <body style="text-align: center; margin-top: 50px; font-family: Arial;">
        <h1>🚀 Welcome to Cybersecurity Prediction API!</h1>
        <p>Use POST request at <code>/predict</code> to get predictions.</p>
      </body>
    </html>
    '''

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    input_features = [data['feature1'], data['feature2'], data['feature3']]
    final_features = scaler.transform([input_features])
    prediction = model.predict(final_features)

    prediction_class = int(prediction[0])
    label_dict = {0: "Normal", 1: "DoS", 2: "Phishing", 3: "Malware", 4: "Other"}
    prediction_label = label_dict.get(prediction_class, "Unknown")

    return jsonify({
        'prediction_class': prediction_class,
        'prediction_label': prediction_label
    })

if __name__ == '__main__':
    app.run(debug=True)
