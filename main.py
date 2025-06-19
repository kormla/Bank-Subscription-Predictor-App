# main.py
import flask
from flask import request, jsonify, Flask, render_template # Added render_template
import joblib
import pandas as pd
import numpy as np
import warnings
import traceback

warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# --- Load the trained model and features ---
# Make sure these paths are correct relative to your main.py
model = None
model_features = None
try:
    model = joblib.load('random_forest_model.pkl')
    # Load the list of expected feature columns from training
    model_features = joblib.load('model_features.pkl')
    print("Model and features loaded successfully.")
except Exception as e:
    print(f"Error loading model or features: {e}")

# --- Define Preprocessing Function (Crucial for consistent predictions) ---
def preprocess_input(data: dict, model_features: list) -> pd.DataFrame:
    """
    Applies the same preprocessing steps as performed during model training.
    """
    # Create a DataFrame from the input dictionary
    df_input = pd.DataFrame([data])

    # --- 1. Replicate 'unknown' handling (if applicable for new inputs) ---
    # In a real API, you might validate input or assume clean input.
    # For now, we'll assume 'unknown' are not directly sent, or need specific handling.
    # If your model needs to handle 'unknown' in new data, you'd replicate that logic here.
    # For simplicity, we assume the input dictionary already reflects cleaned categorical values.

    # --- 2. Replicate dropping 'duration' (if present in input) ---
    if 'duration' in df_input.columns:
        df_input = df_input.drop('duration', axis=1)

    # --- 3. Replicate Feature Engineering ---
    if 'pdays' in df_input.columns:
        df_input['was_contacted_before'] = (df_input['pdays'] != -1).astype(int)
    else:
        # Handle case where pdays might be missing in new input (e.g., default to 0)
        df_input['was_contacted_before'] = 0

    if 'campaign' in df_input.columns:
        df_input['multiple_campaign_contacts'] = (df_input['campaign'] > 1).astype(int)
    else:
        # Handle case where campaign might be missing in new input
        df_input['multiple_campaign_contacts'] = 0

    # --- 4. Replicate Outlier Handling (Capping) ---
    # IMPORTANT: You need to load the actual Q1, Q3, and IQR values (or the calculated bounds)
    # from your training data and use them here. Hardcoding is not robust.
    # For example, if you saved these bounds in a dictionary, you'd load them here.
    numerical_cols_for_outlier_handling = ['age', 'balance', 'campaign', 'pdays', 'previous', 'day']
    for col in numerical_cols_for_outlier_handling:
        if col in df_input.columns and col in model_features:
            # PLACEHOLDER: You NEED to load your actual calculated lower_bound and upper_bound for each column
            # from your training data and replace these hardcoded zeros.
            # For demonstration, we'll use simple hardcoded values that might not be correct for your model:
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


    # --- 5. Replicate One-Hot Encoding ---
    # Identify categorical columns that were originally one-hot encoded
    # This list MUST exactly match what was used during training.
    original_categorical_cols = [
        'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
        'month', 'poutcome'
    ]

    # Convert object columns in current input to category type for consistency with get_dummies
    for col in original_categorical_cols:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype('category')

    # Apply one-hot encoding to the input DataFrame
    df_processed = pd.get_dummies(df_input, columns=original_categorical_cols, drop_first=True)

    # --- CRITICAL STEP: Replicate column name cleaning from notebook (after OHE) ---
    # The notebook's training process applied this to X.columns after get_dummies.
    # It ensures that generated names like 'job_blue-collar' become 'job_bluecollar'.
    df_processed.columns = df_processed.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)

    # --- Align columns with the model's training features (CRITICAL STEP) ---
    # This ensures that your input DataFrame for prediction has the exact same columns
    # in the exact same order as your training data.
    missing_cols = set(model_features) - set(df_processed.columns)
    for c in missing_cols:
        df_processed[c] = 0 # Add missing dummy variables as 0

    # Ensure the order of columns is the same as the training data
    df_final = df_processed[model_features]

    return df_final

# --- Home Endpoint ---
@app.route("/", methods=['GET']) # Added methods=['GET']
def home():
    # Render the HTML form
    return render_template("index.html")

# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or model_features is None:
        # For web form, render error on page; for API, return JSON
        if request.is_json:
            return jsonify({'error': 'Model or features not loaded. Check server logs.'}), 500
        else:
            return render_template("index.html", error_message='Model or features not loaded. Check server logs.')

    # Determine if input is JSON (API call) or form data (web form)
    if request.is_json:
        raw_data = request.get_json(force=True)
    else:
        raw_data = request.form # Get data from form submission

    if not raw_data:
        if request.is_json:
            return jsonify({'error': 'No JSON data provided.'}), 400
        else:
            return render_template("index.html", error_message='No data provided.')

    try:
        # Convert form data strings to appropriate types if it's form data
        if not request.is_json:
            processed_raw_data = {}
            # List of numerical features (adjust based on your actual model features)
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

# --- Health Check Endpoint (Optional but good practice) ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200

# --- Main entry point for Heroku / Render ---
if __name__ == '__main__':
    # For local testing:
    # Ensure you have your model and model_features.pkl in the same directory
    # Run with: python main.py (or app.py if renamed)
    # Test with example curl command provided in the original notebook
    app.run(debug=True, host='0.0.0.0', port=5000)
