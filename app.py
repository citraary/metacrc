import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    with open('metastasis_rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load model
model = load_model()

# App title and description
st.title("Metastasis Prediction Tool")
st.markdown("""
Predict the likelihood of metastasis based on snail and emt expression levels.

**Levels:**
- **1**: Low expression
- **2**: Moderate expression  
- **3**: High expression
""")

# Input sliders
st.header("Input Parameters")
col1, col2 = st.columns(2)

with col1:
    snail = st.slider("Snail Expression Level", 1, 3, 2, 
                     help="1=Low, 2=Moderate, 3=High")

with col2:
    emt = st.slider("EMT Expression Level", 1, 3, 2,
                   help="1=Low, 2=Moderate, 3=High")

# Prediction
if st.button("Predict Metastasis Risk"):
    # Prepare input data
    input_data = pd.DataFrame({
        'snail': [snail],
        'emt': [emt]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Display results
    st.subheader("Prediction Results")
    
    if prediction == 1:
        st.success("**Prediction: No Metastasis**")
        risk_level = "Low Risk"
        color = "green"
    else:
        st.error("**Prediction: Metastasis Detected**")
        risk_level = "High Risk"
        color = "red"
    
    # Confidence score
    confidence = max(probability) * 100
    st.metric("Confidence Score", f"{confidence:.1f}%")
    
    # Probability breakdown
    st.subheader("Probability Breakdown")
    prob_df = pd.DataFrame({
        'Outcome': ['No Metastasis', 'Metastasis'],
        'Probability': [f"{probability[0]*100:.1f}%", f"{probability[1]*100:.1f}%"]
    })
    st.dataframe(prob_df, hide_index=True)
    
    # Interpretation
    st.subheader("Interpretation")
    if prediction == 1:
        st.info("The model predicts low risk of metastasis based on the input expression levels.")
    else:
        st.warning("The model predicts high risk of metastasis. Consider further clinical evaluation.")

# Add some information about the model
with st.expander("About this Model"):
    st.markdown("""
    **Model Details:**
    - Algorithm: Random Forest Classifier
    - Training Data: 41 samples with snail/emt expression levels
    - Target: Metastasis prediction (1=No, 2=Yes)
    
    **Note:** This is a demonstration model. For clinical use, 
    consult with healthcare professionals and validate with larger datasets.
    """)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn")