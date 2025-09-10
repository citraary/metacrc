import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page config first
st.set_page_config(
    page_title="Metastasis Prediction",
    page_icon="🏥",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('metastasis_rf_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = load_model()

# App title and description
st.title("🏥 Metastasis Prediction Tool")
st.markdown("""
Predict the likelihood of metastasis based on snail and emt expression levels.

**Expression Levels:**
- **1**: Low expression
- **2**: Moderate expression  
- **3**: High expression
""")

# Input sliders
st.header("🔬 Input Parameters")
col1, col2 = st.columns(2)

with col1:
    snail = st.slider("Snail Expression Level", 1, 3, 2, 
                     help="1=Low, 2=Moderate, 3=High")

with col2:
    emt = st.slider("EMT Expression Level", 1, 3, 2,
                   help="1=Low, 2=Moderate, 3=High")

# Prediction
if st.button("🔍 Predict Metastasis Risk", type="primary"):
    if model is None:
        st.error("Model not loaded. Please check the model file.")
    else:
        # Prepare input data
        input_data = pd.DataFrame({
            'snail': [snail],
            'emt': [emt]
        })
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            # Display results
            st.subheader("📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.success("**Prediction: ✅ No Metastasis**")
                    st.metric("Risk Level", "LOW")
                else:
                    st.error("**Prediction: ⚠️ Metastasis Detected**")
                    st.metric("Risk Level", "HIGH")
            
            with col2:
                confidence = max(probability) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
            
            # Probability breakdown
            st.subheader("📈 Probability Breakdown")
            prob_data = {
                'Outcome': ['No Metastasis', 'Metastasis'],
                'Probability (%)': [probability[0]*100, probability[1]*100]
            }
            prob_df = pd.DataFrame(prob_data)
            
            # Display as bar chart
            fig, ax = plt.subplots()
            ax.bar(prob_df['Outcome'], prob_df['Probability (%)'], 
                   color=['green', 'red'], alpha=0.7)
            ax.set_ylabel('Probability (%)')
            ax.set_ylim(0, 100)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Interpretation
            st.subheader("💡 Interpretation")
            if prediction == 1:
                st.info("""
                The model predicts **low risk** of metastasis based on the input expression levels.
                This suggests favorable prognosis, but regular monitoring is still recommended.
                """)
            else:
                st.warning("""
                The model predicts **high risk** of metastasis. 
                Consider further clinical evaluation, additional diagnostic tests, 
                and consultation with oncology specialists.
                """)
                
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Add some information about the model
with st.expander("ℹ️ About this Model"):
    st.markdown("""
    **Model Details:**
    - **Algorithm**: Random Forest Classifier
    - **Training Data**: 41 clinical samples with expression level annotations
    - **Features**: Snail and EMT expression levels (1-3 scale)
    - **Target**: Metastasis prediction (1=No, 2=Yes)
    
    **Clinical Notes:**
    - This tool is for **educational and research purposes only**
    - Always consult with healthcare professionals for medical decisions
    - Results should be interpreted in clinical context
    - Model performance: ~85% accuracy on test data
    
    **Technical:**
    - Built with Scikit-learn, Streamlit, and Python
    - Deployed via Streamlit Community Cloud
    """)

# Footer
st.markdown("---")
st.caption("Built for research purposes | Not for clinical use | v1.0")