from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model
# In Vercel, the current working directory is the project root or api folder depending on setup
# We will use __file__ to find the model relative to this script
model_path = os.path.join(os.path.dirname(__file__), 'rf_model.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded properly.'}), 500

    data = request.json
    if not data or 'features' not in data:
        return jsonify({'error': 'No features provided.'}), 400
    
    features = data['features']
    
    if len(features) != 22:
        return jsonify({'error': f'Expected 22 features, got {len(features)}'}), 400

    try:
        # Convert to numpy array and reshape for a single prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Predict
        prediction = model.predict(features_array)
        
        return jsonify({
            'prediction': int(prediction[0]),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Vercel requires the app to be exported, but executing it locally is fine too
if __name__ == '__main__':
    app.run(debug=True, port=5000)
