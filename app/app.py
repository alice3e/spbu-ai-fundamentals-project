from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import re
import joblib
import json
import category_encoders as ce # Make sure this is in requirements

app = Flask(__name__)

# --- LOAD ALL PRE-FITTED OBJECTS AND DATA ---
model = joblib.load('saved_model/model.pkl')
scaler = joblib.load('saved_model/scaler.pkl')
num_imputer = joblib.load('saved_model/num_imputer.pkl')
cat_imputer = joblib.load('saved_model/cat_imputer.pkl')
target_encoder = joblib.load('saved_model/target_encoder.pkl')

with open('saved_model/ohe_columns.json', 'r') as f:
    ohe_columns = json.load(f)

with open('saved_model/experience_counts.json', 'r') as f:
    experience_counts = json.load(f)
    
# Load a small part of the data or reference data for fitting some transformers if needed
# For experience counts, we loaded them from JSON.
# For target encoder, it's already fitted. For one-hot, we have column names.
# If any imputer or encoder was fit on 'train_data' in notebook, that specific version
# of train_data (or its characteristics) should be available or the object pre-fitted.
# For simplicity now, we'll assume the pickled objects are sufficient.
# If 'distributor' or 'director' modes for imputation were derived from 'train_data',
# we might need a reference for that specific mode, or ensure cat_imputer handles it.

# Columns for different encoding strategies (must match notebook)
columns_for_target_encoding = ['main_actor_4', 'main_actor_3', 'writer', 'main_actor_2', 'producer', 'director', 'main_actor_1', 'cinematographer', 'composer', 'distributor']
columns_for_one_hot = ['genre_1', 'genre_2', 'genre_3', 'genre_4', 'mpaa']


def convert_runtime_to_minutes(value):
    if pd.isna(value):
        return None # Let imputer handle it
    match = re.match(r'(?:(\d+)\s*hr)?\s*(?:(\d+)\s*min)?', str(value))
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes
    return None

def preprocess_input(input_data_df):
    # df = pd.DataFrame([input_data_dict]) # Already a DataFrame
    df = input_data_df.copy()

    # Feature Engineering: Experience (using pre-loaded counts)
    for col_name in ["director", "writer", "producer", "composer", "cinematographer"]:
        df[f"{col_name}_experience"] = df[col_name].map(experience_counts.get(col_name, {})).fillna(0)
    for i in range(1, 5):
        actor_col = f"main_actor_{i}"
        df[f"{actor_col}_experience"] = df[actor_col].map(experience_counts.get(actor_col, {})).fillna(0)
    
    df["cast_popularity"] = sum(df[f"main_actor_{i}_experience"] for i in range(1, 5))

    # Convert run_time
    df['run_time'] = df['run_time'].apply(convert_runtime_to_minutes)

    # Impute Numerical
    num_cols_to_impute = df.select_dtypes(include=[np.number]).columns.tolist()
    # Ensure 'worldwide' is not imputed if it's accidentally passed or part of num_cols logic
    if 'worldwide' in num_cols_to_impute:
        num_cols_to_impute.remove('worldwide')
    df[num_cols_to_impute] = num_imputer.transform(df[num_cols_to_impute])
    
    # Impute Categorical (those not handled by target/OHE specifically if any, or fill specific ones)
    # Note: The notebook logic for specific imputation (groupby director) is complex to replicate
    # exactly without the training data context for modes.
    # Using the saved cat_imputer which was fitted on the training data is more robust.
    cat_cols_for_general_imputation = df.select_dtypes(include=['object']).columns.tolist()
    # Remove columns handled by target/OHE or specific filling
    cols_handled_elsewhere = set(columns_for_target_encoding + columns_for_one_hot)
    cat_cols_for_general_imputation = [c for c in cat_cols_for_general_imputation if c not in cols_handled_elsewhere]
    
    if cat_cols_for_general_imputation: # If there are any such columns
         df[cat_cols_for_general_imputation] = cat_imputer.transform(df[cat_cols_for_general_imputation])


    # Fill specific NaNs for genres and main_actor_4 before OHE/Target Encoding
    # if they weren't covered by the general cat_imputer (they should be if it was fit on all object columns)
    genre_actor_fillna_cols = ['genre_1', 'genre_2', 'genre_3', 'genre_4', 'main_actor_4', 'mpaa', 'distributor']
    for col in genre_actor_fillna_cols:
        if col in df.columns: # Ensure column exists
             # If cat_imputer was trained on these, this fillna is redundant
             # If not, this 'Unknown' fill is crucial
            df[col] = df[col].fillna('Unknown')


    # One-Hot Encoding
    df = pd.get_dummies(df, columns=columns_for_one_hot, prefix=columns_for_one_hot)
    
    # Align columns with training data OHE columns
    # Add missing columns (that were in training but not in this input) and fill with 0
    for col in ohe_columns:
        if col not in df.columns and any(prefix in col for prefix in [f"{c}_" for c in columns_for_one_hot]):
            df[col] = 0
            
    # Target Encoding (use the pre-fitted encoder)
    # Ensure target encoder only sees columns it was trained on
    cols_for_te_in_df = [col for col in columns_for_target_encoding if col in df.columns]
    if cols_for_te_in_df:
         df[cols_for_te_in_df] = target_encoder.transform(df[cols_for_te_in_df])


    # Ensure final column order and presence matches `ohe_columns`
    # `ohe_columns` should be the list of columns *after* all preprocessing (OHE, target encoding, feature eng)
    # but *before* scaling from your notebook's X.
    # This step is CRITICAL.
    df = df.reindex(columns=ohe_columns, fill_value=0)

    # Scaling (use the pre-fitted scaler)
    df_scaled = scaler.transform(df)
    
    return df_scaled


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data_json = request.get_json(force=True)
        app.logger.info(f"Received data: {data_json}")

        # Convert input dictionary to DataFrame
        # Handle potential missing fields by setting them to NaN
        # List all expected feature names your model uses BEFORE preprocessing
        expected_features = [
            'movie_year', 'director', 'writer', 'producer', 'composer',
            'cinematographer', 'main_actor_1', 'main_actor_2', 'main_actor_3',
            'main_actor_4', 'budget', 'domestic', 'international',
            'mpaa', 'run_time', 'genre_1', 'genre_2', 'genre_3', 'genre_4', 'distributor'
        ]
        
        input_dict_prepared = {feat: data_json.get(feat, np.nan) for feat in expected_features}
        input_df = pd.DataFrame([input_dict_prepared])

        # Convert types for numerical columns before preprocessing
        # (assuming they come as strings from JSON if not explicitly numbers)
        num_potentially_string_cols = ['movie_year', 'budget', 'domestic', 'international']
        for col in num_potentially_string_cols:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')


        processed_features = preprocess_input(input_df)
        
        prediction = model.predict(processed_features)
        
        # If you log-transformed 'worldwide' during training, you need to inverse transform here
        # prediction_original_scale = np.expm1(prediction)
        # For now, assuming no log transform on target in the snippet.
        prediction_original_scale = prediction

        app.logger.info(f"Prediction: {prediction_original_scale[0]}")
        return jsonify({'prediction': round(prediction_original_scale[0], 2)})
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)