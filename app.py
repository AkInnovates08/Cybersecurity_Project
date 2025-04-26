from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define mapping of predictions to labels
label_mapping = {
    0: "Benign",
    1: "Malware",
    2: "Phishing",
    3: "Ransomware",
    4: "Other Threat"
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Example assuming 3 features
    features = np.array([data['feature1'], data['feature2'], data['feature3']]).reshape(1, -1)
    
    # Scale input features
    scaled_features = scaler.transform(features)
    
    # Predict
    prediction = model.predict(scaled_features)
    predicted_class = int(prediction[0])
    
    # Map the prediction to the label
    label = label_mapping.get(predicted_class, "Unknown")
    
    return jsonify({
        'prediction_class': predicted_class,
        'prediction_label': label
    })

if __name__ == '__main__':
    app.run(debug=True)
