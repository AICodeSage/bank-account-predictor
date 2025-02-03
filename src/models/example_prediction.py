
# Example code for making predictions
import pandas as pd
import joblib

def predict_bank_account(data):
    '''
    data should be a dictionary or DataFrame with columns:
    - household_size
    - age_of_respondent
    - year
    - location_type
    - cellphone_access
    - gender_of_respondent
    '''
    # Load model and preprocessors
    model = joblib.load("models/model.pkl")
    location_encoder = joblib.load("models/location_encoder.pkl")
    cellphone_encoder = joblib.load("models/cellphone_encoder.pkl")
    gender_encoder = joblib.load("models/gender_encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    
    # Convert to DataFrame if dictionary
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    # Preprocess input
    data['location_type'] = location_encoder.transform(data['location_type'])
    data['cellphone_access'] = cellphone_encoder.transform(data['cellphone_access'])
    data['gender_of_respondent'] = gender_encoder.transform(data['gender_of_respondent'])
    
    # Create dummy columns
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0
    
    # Ensure columns are in the same order
    data = data[feature_columns]
    
    # Scale features
    data_scaled = scaler.transform(data)
    
    # Make prediction
    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)
    
    return prediction[0], probability[0]

# Example usage:
sample_input = {
    'household_size': 4,
    'age_of_respondent': 25,
    'year': 2024,
    'location_type': 'Urban',
    'cellphone_access': 'Yes',
    'gender_of_respondent': 'Female'
}

prediction, probability = predict_bank_account(sample_input)
print(f"Prediction: {'Has bank account' if prediction == 1 else 'No bank account'}")
print(f"Probability: {probability[1]:.2%}")
