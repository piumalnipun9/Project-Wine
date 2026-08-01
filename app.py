import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and scaler
@st.cache_resource
def load_models():
    # Note: paths assume app.py is in the root directory
    model = joblib.load('models/random_forest_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    return model, scaler

try:
    model, scaler = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Error loading models: {e}. Please ensure you have run the training script first.")

st.title("🍷 Wine Quality Prediction System")
st.markdown("""
This application predicts whether a wine is of **Good Quality** (score $\ge$ 6) or **Bad Quality** (score $<$ 6) based on its physicochemical properties.
""")

if models_loaded:
    st.sidebar.header("Input Features")
    
    # Define inputs for all 11 features
    fixed_acidity = st.sidebar.slider("Fixed Acidity", min_value=4.0, max_value=16.0, value=7.4, step=0.1)
    volatile_acidity = st.sidebar.slider("Volatile Acidity", min_value=0.1, max_value=1.6, value=0.5, step=0.01)
    citric_acid = st.sidebar.slider("Citric Acid", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    residual_sugar = st.sidebar.slider("Residual Sugar", min_value=0.5, max_value=16.0, value=2.0, step=0.1)
    chlorides = st.sidebar.slider("Chlorides", min_value=0.01, max_value=0.6, value=0.08, step=0.001)
    free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", min_value=1.0, max_value=72.0, value=15.0, step=1.0)
    total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", min_value=6.0, max_value=290.0, value=46.0, step=1.0)
    density = st.sidebar.slider("Density", min_value=0.99, max_value=1.005, value=0.996, step=0.0001, format="%.4f")
    pH = st.sidebar.slider("pH", min_value=2.5, max_value=4.1, value=3.3, step=0.01)
    sulphates = st.sidebar.slider("Sulphates", min_value=0.3, max_value=2.0, value=0.65, step=0.01)
    alcohol = st.sidebar.slider("Alcohol (%)", min_value=8.0, max_value=15.0, value=10.5, step=0.1)
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    })
    
    st.subheader("Your Input:")
    st.dataframe(input_data)
    
    if st.button("Predict Quality"):
        # Scale inputs
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        st.subheader("Prediction Result:")
        if prediction == 1:
            st.success(f"🌟 **Good Quality Wine!** (Probability: {probability:.2%})")
        else:
            st.error(f"📉 **Bad Quality Wine.** (Probability of being good: {probability:.2%})")
            
    st.markdown("---")
    st.markdown("**Note:** This model uses a Random Forest Classifier trained on the WineQT dataset.")
