"""
Local development server - serves both the 3D frontend and the Python ML API.
Run this file and open http://localhost:5000 in your browser.
"""
from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder='.', static_url_path='')

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'api', 'rf_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print(f"[OK] Model loaded successfully from {model_path}")

# Serve the frontend
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

# API endpoint
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'features' not in data:
        return jsonify({'error': 'No features provided.'}), 400

    features = data['features']
    if len(features) != 22:
        return jsonify({'error': f'Expected 22 features, got {len(features)}'}), 400

    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)

    return jsonify({
        'prediction': int(prediction[0]),
        'status': 'success'
    })

if __name__ == '__main__':
    print("\nParkinson's Disease Detection - Local Server")
    print("=" * 50)
    print("Open your browser at: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)
