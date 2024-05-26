import warnings
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Load the trained model
model = joblib.load('best_random_forest_model.pkl')




def predict_zone(age, body_temp, heart_rate, patient_id, gender_encoded, age_group_encoded):
    # Make predictions based on input features
    prediction = model.predict([[age, body_temp, heart_rate, patient_id, gender_encoded, age_group_encoded]])
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        body_temp = float(request.form['body_temp'])
        heart_rate = float(request.form['heart_rate'])
        patient_id = float(request.form['patient_id'])
        gender_encoded = int(request.form['gender_encoded'])
        age_group_encoded = int(request.form['age_group_encoded'])
        
        # Predict temperature zone using all features
        zone = predict_zone(age, body_temp, heart_rate, patient_id, gender_encoded, age_group_encoded)
        
        return render_template('result.html', temperature=body_temp, zone=zone)



if __name__ == '__main__':
    app.run(debug=True)
