import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_dir = os.path.join(project_root, "Data")
models_dir = os.path.join(current_dir, "models")

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# Load datasets
try:
    train = pd.read_csv(os.path.join(data_dir, "Train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "Test.csv"))
    print("Data loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: Could not find data files in {data_dir}")
    raise e

# Create encoders and scaler
location_encoder = LabelEncoder()
cellphone_encoder = LabelEncoder()
gender_encoder = LabelEncoder()

# Convert target
target_encoder = LabelEncoder()
train['bank_account'] = target_encoder.fit_transform(train['bank_account'])

# Enhanced preprocessing
def preprocess_data(data, is_training=True):
    data = data.copy()
    
    # Convert numerical columns
    num_cols = ["household_size", "age_of_respondent", "year"]
    data[num_cols] = data[num_cols].astype(float)
    
    # Feature engineering
    data['age_group'] = pd.cut(data['age_of_respondent'], 
                              bins=[0, 25, 35, 50, 100], 
                              labels=['Young', 'Adult', 'Middle_Aged', 'Senior'])
    
    # Encode categorical features
    categorical_cols = ["relationship_with_head", "marital_status", "education_level", 
                       "job_type", "country", "age_group"]
    data = pd.get_dummies(data, columns=categorical_cols, prefix_sep="_")
    
    if is_training:
        feature_columns = data.columns.tolist()
        joblib.dump(feature_columns, os.path.join(models_dir, "feature_columns.pkl"))
    
    # Label encode binary variables
    if is_training:
        data["location_type"] = location_encoder.fit_transform(data["location_type"])
        data["cellphone_access"] = cellphone_encoder.fit_transform(data["cellphone_access"])
        data["gender_of_respondent"] = gender_encoder.fit_transform(data["gender_of_respondent"])
    else:
        data["location_type"] = location_encoder.transform(data["location_type"])
        data["cellphone_access"] = cellphone_encoder.transform(data["cellphone_access"])
        data["gender_of_respondent"] = gender_encoder.transform(data["gender_of_respondent"])
    
    # Drop ID if present
    if "uniqueid" in data.columns:
        data.drop(columns=["uniqueid"], inplace=True)
    
    return data

# Process data
X = train.drop(columns=['bank_account'])
y = train['bank_account']

# Get preprocessed data
X_processed = preprocess_data(X, is_training=True)
feature_names = X_processed.columns.tolist()

# Split data with stratification
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Train model with balanced weights
print("Training model...")
model = XGBClassifier(
    max_depth=6,
    learning_rate=0.01,
    n_estimators=1000,
    random_state=42,
    scale_pos_weight=class_weights[1]/class_weights[0]  # Weight for positive class
)

# Train the model
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_val_scaled)
y_pred_proba = model.predict_proba(X_val_scaled)

# Print detailed metrics
print("\nModel Performance:")
print("Classification Report:")
print(classification_report(y_val, y_pred))

# Plot feature importance
plt.figure(figsize=(12, 6))
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.savefig(os.path.join(models_dir, 'feature_importance.png'))

# Calculate and print validation metrics
val_accuracy = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {val_accuracy:.4f}")

# Calculate class distribution
class_distribution = pd.Series(y_train).value_counts(normalize=True)
print("\nClass Distribution in Training Data:")
print(class_distribution)

# Save components
print("\nSaving model files...")
joblib.dump(model, os.path.join(models_dir, "model.pkl"))
joblib.dump(location_encoder, os.path.join(models_dir, "location_encoder.pkl"))
joblib.dump(cellphone_encoder, os.path.join(models_dir, "cellphone_encoder.pkl"))
joblib.dump(gender_encoder, os.path.join(models_dir, "gender_encoder.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
joblib.dump(target_encoder, os.path.join(models_dir, "target_encoder.pkl"))
joblib.dump(feature_names, os.path.join(models_dir, "feature_columns.pkl"))

print("Training complete! Model files saved in:", models_dir)

# Save example prediction code
example_code = """
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
"""

with open(os.path.join(models_dir, "example_prediction.py"), "w") as f:
    f.write(example_code)

print("\nExample prediction code saved to:", os.path.join(models_dir, "example_prediction.py")) 