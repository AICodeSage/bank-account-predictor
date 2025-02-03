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
st.markdown("""
<style>
    /* Modern color scheme */
    :root {
        --primary-color: #4A90E2;
        --secondary-color: #2ECC71;
        --text-color: #2C3E50;
        --error-color: #E74C3C;
    }
    
    /* Global styles */
    .main {
        background: transparent;
        padding: 2rem;
        color: var(--text-color);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        height: 3.5em;
        background: linear-gradient(45deg, rgba(74, 144, 226, 0.9), rgba(46, 204, 113, 0.9));
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    /* Card styling */
    .prediction-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        color: #666666;  /* Gray text */
    }
    
    /* Success/Error boxes */
    .success-box {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.9), rgba(39, 174, 96, 0.9));
        color: #666666;  /* Gray text */
    }
    .error-box {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.9), rgba(192, 57, 43, 0.9));
        color: #666666;  /* Gray text */
    }
    
    /* Info box styling */
    .info-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        color: #666666;  /* Gray text */
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(45deg, #4A90E2, #2ECC71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem;
        font-size: 2.5em;
        font-weight: 800;
    }
    h3 {
        color: var(--primary-color);
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    h4 {
        color: var(--text-color);
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, #4A90E2, #2ECC71);
    }
    
    /* Form inputs */
    .stSelectbox, .stSlider {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Lists in info box */
    .info-box ul {
        list-style-type: none;
        padding-left: 0;
    }
    .info-box li {
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
        color: #666666;
    }
    .info-box li:last-child {
        border-bottom: none;
    }
    
    /* Footer */
    .footer {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        box-shadow: 0 -5px 15px rgba(0,0,0,0.05);
    }
    
    /* Update text colors */
    .success-box, .error-box {
        color: #666666;  /* Gray text */
    }
    
    .info-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        color: #666666;  /* Gray text */
    }
    
    /* Form labels */
    .stSelectbox label, .stSlider label {
        color: #666666 !important;  /* Gray text */
    }
    
    /* Button text */
    .stButton>button {
        color: #666666;  /* Gray text */
    }
    
    /* Update all paragraph text */
    p {
        color: #666666 !important;
    }
    
    /* Center-align form elements */
    .stForm {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Form inputs styling */
    .stSelectbox, .stSlider {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Center text in form */
    .stForm label {
        text-align: center;
        display: block;
    }
    
    /* Improve spacing */
    .stForm > div {
        margin-bottom: 1.5rem;
    }
    
    /* Enhanced text visibility */
    h1, h2, h3, h4, p, label, .stMarkdown {
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        color: #2C3E50 !important;  /* Darker gray for better contrast */
        font-weight: 500;
    }
    
    /* Make form labels more visible */
    .stSelectbox label, .stSlider label {
        color: #2C3E50 !important;
        font-size: 1.1em;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    /* Enhance info box text */
    .info-box {
        color: #2C3E50 !important;
        font-size: 1.1em;
    }
    
    .info-box h4 {
        font-size: 1.3em;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .info-box li {
        color: #2C3E50 !important;
        font-weight: 500;
        padding: 0.7rem 0;
    }
    
    /* Make prediction results more visible */
    .prediction-box {
        font-size: 1.2em;
    }
    
    .prediction-box h3 {
        font-size: 1.5em;
        font-weight: 600;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Enhance form text */
    .stForm {
        font-size: 1.1em;
    }
    
    /* Make button text more visible */
    .stButton>button {
        color: #2C3E50 !important;
        font-size: 1.2em;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.2);
    }
    
    /* Footer text enhancement */
    .footer {
        color: #2C3E50 !important;
        font-size: 1.1em;
        font-weight: 500;
    }
    
    /* Enhance gauge chart text */
    .js-plotly-plot text {
        font-weight: 500 !important;
        font-size: 1.1em !important;
    }
    
    /* Make all text white with better visibility */
    h1, h2, h3, h4, p, label, .stMarkdown {
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        color: white !important;
        font-weight: 500;
    }
    
    /* Form labels */
    .stSelectbox label, .stSlider label {
        color: white !important;
        font-size: 1.1em;
        font-weight: 500;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Info box text */
    .info-box {
        color: white !important;
        font-size: 1.1em;
    }
    
    .info-box h4 {
        color: white !important;
        font-size: 1.3em;
        font-weight: 600;
    }
    
    .info-box li {
        color: white !important;
        font-weight: 500;
    }
    
    /* Button text */
    .stButton>button {
        color: white !important;
        font-size: 1.2em;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Footer text */
    .footer {
        color: white !important;
    }
    
    .footer span {
        color: white !important;
    }
    
    /* Gauge chart text */
    .js-plotly-plot text {
        fill: white !important;
        font-weight: 500 !important;
        font-size: 1.1em !important;
    }
    
    /* Streamlit selects and inputs */
    .stSelectbox div[data-baseweb="select"] span {
        color: white !important;
    }
    
    .stSlider [data-testid="stMarkdownContainer"] {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

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