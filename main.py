<<<<<<< HEAD
# app.py
import flask
from flask import request, jsonify, Flask
import joblib
import pandas as pd
import numpy as np
import warnings
import traceback # Import the traceback module

warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# --- Load the trained model and features ---
# Make sure these paths are correct relative to your app.py
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
    # numerical_cols_for_outlier_handling = ['balance', 'campaign', 'pdays', 'previous']
    # for col in numerical_cols_for_outlier_handling:
    #     if col in df_input.columns and col in model_features:
    #         lower_bound = # Load lower bound for 'col'
    #         upper_bound = # Load upper bound for 'col'
    #         df_input[col] = np.where(df_input[col] < lower_bound, lower_bound, df_input[col])
    #         df_input[col] = np.where(df_input[col] > upper_bound, upper_bound, df_input[col])


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

    # Align columns with the model's training features (CRITICAL STEP)
    # This ensures that your input DataFrame for prediction has the exact same columns
    # in the exact same order as your training data.
    missing_cols = set(model_features) - set(df_processed.columns)
    for c in missing_cols:
        df_processed[c] = 0 # Add missing dummy variables as 0

    # Ensure the order of columns is the same as the training data
    df_final = df_processed[model_features]

    return df_final

# --- Home Endpoint ---
@app.route("/")
def home():
    return "Hello, this is my prediction app!"

# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or model_features is None:
        return jsonify({'error': 'Model or features not loaded. Check server logs.'}), 500

    try:
        # Get data from the POST request
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No JSON data provided.'}), 400

        # Preprocess the input data
        processed_data = preprocess_input(data, model_features)

        # Make prediction
        prediction_proba = model.predict_proba(processed_data)[:, 1][0] # Probability of 'yes'
        prediction = model.predict(processed_data)[0] # 0 or 1

        # Return the prediction as JSON
        result = {
            'prediction': int(prediction), # Convert numpy int to Python int for JSON
            'probability_of_subscription': float(prediction_proba) # Convert numpy float to Python float
        }
        return jsonify(result), 200

    except Exception as e:
        # Catch any errors and return them
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

# --- Health Check Endpoint (Optional but good practice) ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200

# --- Main entry point for Heroku ---
if __name__ == '__main__':
    # For local testing:
    # Ensure you have your model and model_features.pkl in the same directory
    # Run with: python app.py
    # Test with example curl command provided in the original notebook
=======
# app.py
import flask
from flask import request, jsonify, Flask
import joblib
import pandas as pd
import numpy as np
import warnings
import traceback # Import the traceback module

warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# --- Load the trained model and features ---
# Make sure these paths are correct relative to your app.py
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
    # numerical_cols_for_outlier_handling = ['balance', 'campaign', 'pdays', 'previous']
    # for col in numerical_cols_for_outlier_handling:
    #     if col in df_input.columns and col in model_features:
    #         lower_bound = # Load lower bound for 'col'
    #         upper_bound = # Load upper bound for 'col'
    #         df_input[col] = np.where(df_input[col] < lower_bound, lower_bound, df_input[col])
    #         df_input[col] = np.where(df_input[col] > upper_bound, upper_bound, df_input[col])


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

    # Align columns with the model's training features (CRITICAL STEP)
    # This ensures that your input DataFrame for prediction has the exact same columns
    # in the exact same order as your training data.
    missing_cols = set(model_features) - set(df_processed.columns)
    for c in missing_cols:
        df_processed[c] = 0 # Add missing dummy variables as 0

    # Ensure the order of columns is the same as the training data
    df_final = df_processed[model_features]

    return df_final

# --- Home Endpoint ---
@app.route("/")
def home():
    return "Hello, this is my prediction app!"

# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or model_features is None:
        return jsonify({'error': 'Model or features not loaded. Check server logs.'}), 500

    try:
        # Get data from the POST request
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No JSON data provided.'}), 400

        # Preprocess the input data
        processed_data = preprocess_input(data, model_features)

        # Make prediction
        prediction_proba = model.predict_proba(processed_data)[:, 1][0] # Probability of 'yes'
        prediction = model.predict(processed_data)[0] # 0 or 1

        # Return the prediction as JSON
        result = {
            'prediction': int(prediction), # Convert numpy int to Python int for JSON
            'probability_of_subscription': float(prediction_proba) # Convert numpy float to Python float
        }
        return jsonify(result), 200

    except Exception as e:
        # Catch any errors and return them
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

# --- Health Check Endpoint (Optional but good practice) ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200

# --- Main entry point for Heroku ---
if __name__ == '__main__':
    # For local testing:
    # Ensure you have your model and model_features.pkl in the same directory
    # Run with: python app.py
    # Test with example curl command provided in the original notebook
>>>>>>> eee26f2352cdccf155a9db991609d85e356ebab5
    app.run(debug=True, host='0.0.0.0', port=5000)