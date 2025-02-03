import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Bank Account Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved styling
# Add colorful styling with better visibility
st.markdown("""
<style>
    /* Main title and headers */
    .main-title {
        background: linear-gradient(120deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* All text elements */
    .stMarkdown, p, label, span {
        color: white !important;
        font-size: 1.1em;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Form styling */
    .stForm {
        background: rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4) !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 2rem !important;
        font-size: 1.2em !important;
        font-weight: 600 !important;
        border-radius: 25px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        transition: transform 0.2s !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, rgba(78, 205, 196, 0.1), rgba(255, 107, 107, 0.1));
        border: 2px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    /* Prediction boxes */
    .success-box {
        background: linear-gradient(135deg, rgba(46, 213, 115, 0.2), rgba(46, 213, 115, 0.1));
        border: 2px solid rgba(46, 213, 115, 0.3);
    }
    
    .error-box {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.2), rgba(255, 107, 107, 0.1));
        border: 2px solid rgba(255, 107, 107, 0.3);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        margin-top: 2rem;
    }
    
    .footer span {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    
    /* Gauge chart colors */
    .js-plotly-plot .plot-container {
        filter: drop-shadow(0 4px 12px rgba(0,0,0,0.15));
    }
</style>
""", unsafe_allow_html=True)

# Update header with gradient title
# st.markdown("""
# <div style='text-align: center; padding: 2rem;'>
#     <h1 class='main-title'>
#         <span style='font-size: 1.2em; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>🏦</span>
#         Bank Account Predictor
#     </h1>
#     <p style='color: white !important; font-size: 1.3em; margin-top: -1rem;'>
#         Powered by AI & Machine Learning
#     </p>
# </div>
# """, unsafe_allow_html=True)

# Update footer with gradient text


# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")

# Load components
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(models_dir, "model.pkl"))
    location_encoder = joblib.load(os.path.join(models_dir, "location_encoder.pkl"))
    cellphone_encoder = joblib.load(os.path.join(models_dir, "cellphone_encoder.pkl"))
    gender_encoder = joblib.load(os.path.join(models_dir, "gender_encoder.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(models_dir, "feature_columns.pkl"))
    return model, location_encoder, cellphone_encoder, gender_encoder, scaler, feature_columns

try:
    model, location_encoder, cellphone_encoder, gender_encoder, scaler, feature_columns = load_model()
except:
    st.error("Error: Model files not found. Please run train.py first!")
    st.stop()

# Update the header section
st.markdown("""
<div style='text-align: center; padding: 2rem;'>
    <h1 style='font-size: 3em; margin-bottom: 0.5rem;'>
        <span style='color: #FFD700; font-size: 1.2em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>🏦</span>
        <span style='color: blue;'> Bank Account Predictor</span>
    </h1>
    <p style='font-size: 1.2em; color: white; margin-top: -1rem;'>
        Powered by AI & Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='footer'>
    <p style='color: white !important; font-size: 1.2em;'>
        Get to know the predicted likelihood of having a bank account <span style='color: #FF6B6B;'>❤️</span> | 
    </p>
    <div style='display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;'>
        <span>🔒 Secure</span>
        <span>⚡ Fast</span>
        <span>🎯 Accurate</span>
    </div>
</div>
""", unsafe_allow_html=True)


"""..........................................................................................................."""

"""--------------------------------------------------------------------------------------------------------------"""
# Create layout with different column ratios
col1, col2 = st.columns([1, 2])

# Info box in the first (left) column
with col1:
    st.markdown("""
    <div class='info-box' style='margin-top: 2rem;'>
        <h4>🤖 AI-Powered Predictions</h4>
        <p>Our model considers various factors including:</p>
        <ul>
            <li>Age and Demographics</li>
            <li>Location Information</li>
            <li>Household Characteristics</li>
            <li>Communication Access</li>
        </ul>
        <p>The prediction is based on patterns learned from thousands of real cases.</p>
    </div>
    """, unsafe_allow_html=True)

# Personal info form in the second (right) column
with col2:
    st.markdown("""
    <div style='text-align: center; max-width: 600px; margin: 0 auto;'>
        <h3 style='color: #666; margin-bottom: 1rem;'>Enter Your Details</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        # Create two columns within the form
        form_col1, form_col2 = st.columns(2)
        
        with form_col1:
            st.markdown("<h4 style='text-align: center;'>📋 Personal Information</h4>", unsafe_allow_html=True)
            age = st.slider("Age", min_value=18, max_value=100, value=25)
            gender = st.selectbox("Gender", ["Male", "Female"])
            household_size = st.slider("Household Size", min_value=1, max_value=20, value=4)
        
        with form_col2:
            st.markdown("<h4 style='text-align: center;'>📍 Location & Contact</h4>", unsafe_allow_html=True)
            location = st.selectbox("Location Type", ["Rural", "Urban"])
            cellphone = st.selectbox("Has Cellphone Access?", ["Yes", "No"])
            year = st.selectbox("Year", list(range(2020, 2031)), index=4)
        
        # Center-align the submit button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("Predict Likelihood", use_container_width=True)

if submitted:
    # Create progress bar for processing
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
    
    # Prepare input data
    input_data = pd.DataFrame({
        'household_size': [household_size],
        'age_of_respondent': [age],
        'year': [year],
        'location_type': [location],
        'cellphone_access': [cellphone],
        'gender_of_respondent': [gender]
    })
    
    # Add age group
    input_data['age_group'] = pd.cut(input_data['age_of_respondent'], 
                                    bins=[0, 25, 35, 50, 100], 
                                    labels=['Young', 'Adult', 'Middle_Aged', 'Senior'])
    
    # Process input
    input_data['location_type'] = location_encoder.transform(input_data['location_type'])
    input_data['cellphone_access'] = cellphone_encoder.transform(input_data['cellphone_access'])
    input_data['gender_of_respondent'] = gender_encoder.transform(input_data['gender_of_respondent'])
    
    # Create DataFrame with all features from training
    final_input = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Update values for the features we have
    for col in input_data.columns:
        if col in feature_columns:
            final_input[col] = input_data[col]
    
    # Scale features
    input_scaled = scaler.transform(final_input)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Show results with gauge chart
    st.markdown("### 📊 Prediction Results")
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability[1] * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probability of Having a Bank Account", 'font': {'color': "white"}},
        gauge = {
            'axis': {'range': [0, 100], 'tickcolor': "white", 'tickfont': {'color': "white"}},
            'bar': {'color': "#4A90E2"},
            'steps': [
                {'range': [0, 33], 'color': "rgba(255, 255, 255, 0.1)"},
                {'range': [33, 66], 'color': "rgba(255, 255, 255, 0.2)"},
                {'range': [66, 100], 'color': "rgba(255, 255, 255, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': probability[1] * 100
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show prediction result
    if prediction == 1:
        st.markdown("""
        <div class='prediction-box success-box'>
            <h3>✅ High Likelihood of Having a Bank Account</h3>
            <p>This person is likely to have a bank account with {:.1f}% confidence.</p>
        </div>
        """.format(probability[1] * 100), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='prediction-box error-box'>
            <h3>❌ Low Likelihood of Having a Bank Account</h3>
            <p>This person is unlikely to have a bank account with {:.1f}% confidence.</p>
        </div>
        """.format(probability[0] * 100), unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
    <p style='color: #666; font-size: 1.1em;'>
        Built with ❤️ using Streamlit | Bank Account Predictor v1.0
    </p>
    <div style='display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;'>
        <span style='color: #4A90E2;'>🔒 Secure</span>
        <span style='color: #2ECC71;'>⚡ Fast</span>
        <span style='color: #E74C3C;'>🎯 Accurate</span>
    </div>
</div>
""", unsafe_allow_html=True) 