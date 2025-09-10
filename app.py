import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained Random Forest model
try:
    model = joblib.load('metastasis_model.pkl')
    st.success("✅ Model loaded successfully!")
except:
    st.warning("⚠️ Model file not found. Using fallback predictions.")
    model = None

# Streamlit app title
st.title("Metastasis Prediction Tool")

# App description
st.markdown("""
Predict the likelihood of metastasis based on snail and emt expression levels.

**Expression Levels:**
- **1**: Low expression
- **2**: Moderate expression  
- **3**: High expression
""")

# User input form for feature values
col1, col2 = st.columns(2)

with col1:
    snail = st.selectbox("Snail Expression Level", [1, 2, 3], 
                        help="1=Low, 2=Moderate, 3=High")

with col2:
    emt = st.selectbox("EMT Expression Level", [1, 2, 3],
                      help="1=Low, 2=Moderate, 3=High")

# Create the input data as a dataframe
input_data = pd.DataFrame([[snail, emt]],
                          columns=['snail', 'emt'])

# Make the prediction
if st.button("Predict Metastasis Risk", type="primary"):
    if model is not None:
        try:
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            
            # Display prediction result
            st.subheader("📊 Prediction Results")
            
            if prediction == 1:
                st.success("**Prediction: ✅ No Metastasis**")
                st.metric("Risk Level", "LOW")
            else:
                st.error("**Prediction: ⚠️ Metastasis Detected**")
                st.metric("Risk Level", "HIGH")
            
            # Display confidence
            confidence = max(probabilities) * 100
            st.metric("Confidence Score", f"{confidence:.1f}%")
            
            # Probability breakdown
            st.subheader("📈 Probability Breakdown")
            prob_data = {
                'Outcome': ['No Metastasis', 'Metastasis'],
                'Probability': [f"{probabilities[0]*100:.1f}%", f"{probabilities[1]*100:.1f}%"]
            }
            prob_df = pd.DataFrame(prob_data)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            # Fallback to rule-based prediction
            if snail == 3 and emt == 3:
                st.error("⚠️ Fallback Prediction: High Risk (Both markers elevated)")
            else:
                st.success("✅ Fallback Prediction: Low Risk")
    
    else:
        # Rule-based fallback prediction
        st.subheader("📊 Prediction Results (Rule-based)")
        if snail == 3 and emt == 3:
            st.error("**Prediction: ⚠️ Metastasis Detected**")
            st.metric("Risk Level", "HIGH")
            st.info("Both Snail and EMT show high expression levels")
        elif snail >= 2 and emt >= 2:
            st.warning("**Prediction: ⚠️ Moderate Risk**")
            st.metric("Risk Level", "MODERATE")
            st.info("Elevated expression levels detected")
        else:
            st.success("**Prediction: ✅ No Metastasis**")
            st.metric("Risk Level", "LOW")
            st.info("Expression levels within normal range")

    # Clinical recommendations
    st.subheader("💡 Clinical Recommendations")
    if (model and prediction == 2) or (not model and (snail == 3 and emt == 3)):
        st.warning("""
        **For High Risk Cases:**
        - Recommend immediate specialist consultation
        - Consider additional diagnostic imaging
        - Schedule frequent follow-up appointments
        - Discuss potential treatment options
        """)
    elif (model and prediction == 1) or (not model and (snail <= 2 and emt <= 2)):
        st.info("""
        **For Low Risk Cases:**
        - Continue routine monitoring
        - Maintain regular check-ups
        - No immediate intervention needed
        """)
    else:
        st.info("""
        **For Moderate Risk Cases:**
        - Increase monitoring frequency
        - Consider additional tests in 3-6 months
        - Patient education about warning signs
        """)

# Add information section
with st.expander("ℹ️ About This Tool"):
    st.markdown("""
    **Model Information:**
    - **Algorithm**: Random Forest Classifier
    - **Training Data**: 41 clinical samples
    - **Features**: Snail and EMT expression levels
    - **Target**: Metastasis prediction (1=No, 2=Yes)
    
    **Clinical Notes:**
    - This tool is for **research and educational purposes only**
    - Always consult healthcare professionals for medical decisions
    - Results should be interpreted in clinical context
    
    **Technical Details:**
    - Built with Scikit-learn and Streamlit
    - Model accuracy: ~85% on test data
    - Deployed via Streamlit Community Cloud
    """)

# Footer
st.markdown("---")
st.caption("Built for research purposes | Not for clinical decision making | v1.0")
