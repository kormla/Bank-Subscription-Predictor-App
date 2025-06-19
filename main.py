# app.py
import flask
from flask import request, jsonify, Flask, render_template
import joblib
import pandas as pd
import numpy as np
import warnings
import traceback

warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and features
model = None
model_features = None
try:
    model = joblib.load('random_forest_model.pkl')
    # Load the list of expected feature columns from training
    model_features = joblib.load('model_features.pkl')
    print("Model and features loaded successfully.")
except Exception as e:
    print(f"Error loading model or features: {e}")

# Define Preprocessing Function
def preprocess_input(data: dict, model_features: list) -> pd.DataFrame:
    # Create a DataFrame from the input dictionary
    df_input = pd.DataFrame([data])

    # Replicate dropping 'duration'
    if 'duration' in df_input.columns:
        df_input = df_input.drop('duration', axis=1)

    # Replicate Feature Engineering
    if 'pdays' in df_input.columns:
        df_input['was_contacted_before'] = (df_input['pdays'] != -1).astype(int)
    else:
        df_input['was_contacted_before'] = 0

    if 'campaign' in df_input.columns:
        df_input['multiple_campaign_contacts'] = (df_input['campaign'] > 1).astype(int)
    else:
        df_input['multiple_campaign_contacts'] = 0

    # Replicate Outlier Handling (Capping)
    numerical_cols_for_outlier_handling = ['age', 'balance', 'campaign', 'pdays', 'previous', 'day']
    for col in numerical_cols_for_outlier_handling:
        if col in df_input.columns and col in model_features:
            lower_bound = -1000000
            upper_bound = 1000000

            if col == 'age':
                lower_bound = 18
                upper_bound = 90
            elif col == 'balance':
                lower_bound = -5000
                upper_bound = 50000
            elif col == 'campaign':
                lower_bound = 1
                upper_bound = 20
            elif col == 'pdays':
                lower_bound = -1
                upper_bound = 999
            elif col == 'previous':
                lower_bound = 0
                upper_bound = 50
            elif col == 'day':
                lower_bound = 1
                upper_bound = 31

            df_input[col] = np.where(df_input[col] < lower_bound, lower_bound, df_input[col])
            df_input[col] = np.where(df_input[col] > upper_bound, upper_bound, df_input[col])

    # Replicate One-Hot Encoding
    original_categorical_cols = [
        'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
        'month', 'poutcome'
    ]

    for col in original_categorical_cols:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype('category')

    df_processed = pd.get_dummies(df_input, columns=original_categorical_cols, drop_first=True)
    df_processed.columns = df_processed.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)

    # Align columns with the model's training features
    missing_cols = set(model_features) - set(df_processed.columns)
    for c in missing_cols:
        df_processed[c] = 0 

    # Ensure the order of columns is the same as the training data
    df_final = df_processed[model_features]

    return df_final

# Home Endpoint
@app.route("/", methods=['GET'])
def home():
    # Render the HTML form
    return render_template("index.html")

# API Endpoint for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or model_features is None:
        return render_template("index.html", error_message='Model or features not loaded. Check server logs.')

    # Determine if input is JSON (API call) or form data (web form)
    if request.is_json:
        raw_data = request.get_json(force=True)
    else:
        raw_data = request.form # Get data from form submission

    if not raw_data:
        return render_template("index.html", error_message='No data provided.')

    try:
        # Convert form data strings to appropriate types if it's form data
        if not request.is_json:
            processed_raw_data = {}
            # List of numerical features
            numerical_features = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
            for key, value in raw_data.items():
                if key in numerical_features:
                    try:
                        # Attempt to convert to int, then float if int fails
                        processed_raw_data[key] = int(value)
                    except ValueError:
                        processed_raw_data[key] = float(value)
                else:
                    processed_raw_data[key] = value
            data_to_predict = processed_raw_data
        else:
            data_to_predict = raw_data

        # Preprocess the input data
        processed_data = preprocess_input(data_to_predict, model_features)

        # Make prediction
        prediction_proba = model.predict_proba(processed_data)[:, 1][0] # Probability of 'yes'
        prediction = model.predict(processed_data)[0] # 0 or 1

        # Determine readable prediction result
        prediction_text = "subscribe" if prediction == 1 else "not subscribe"

        if request.is_json:
            # Return the prediction as JSON for API calls
            result = {
                'prediction': int(prediction),
                'probability_of_subscription': float(prediction_proba)
            }
            return jsonify(result), 200
        else:
            # Render the form again with the prediction result for web calls
            return render_template("index.html",
                                   prediction_result=prediction_text,
                                   probability_of_subscription=prediction_proba)

    except Exception as e:
        error_info = {'error': str(e), 'trace': traceback.format_exc()}
        print(f"Prediction Error: {error_info}") # Print to server logs
        if request.is_json:
            return jsonify(error_info), 500
        else:
            return render_template("index.html",
                                   error_message=str(e),
                                   error_trace=traceback.format_exc())

# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200

# Main entry point for Render
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
