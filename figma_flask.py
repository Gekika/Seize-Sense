# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and feature names
model = joblib.load('best_model.pkl')
feature_names = joblib.load('feature_names.pkl')

# Preprocessing function (if any preprocessing was applied during training)
def preprocess_input(data):
    # Add preprocessing steps if any were applied during training
    return data

# API endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        input_data = request.json
        input_df = pd.DataFrame([input_data])

        # Ensure the input data has the correct feature names
        input_df = input_df.reindex(columns=feature_names)

        # Preprocess the input data
        processed_data = preprocess_input(input_df)

        # Make predictions
        prediction = model.predict(processed_data)

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
