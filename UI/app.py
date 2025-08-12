import streamlit as st
import pandas as pd
import pickle
import os

# ------------------------
# Configuration and Setup
# ------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ------------------------
# Load trained model & preprocessing objects
# ------------------------
@st.cache_resource
def load_model_components():
    """Load all model components with error handling"""
    try:
        with open('models/final_model_XGBoost_model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        with open('.pkl/columns.pkl', 'rb') as file:
            expected_cols = pickle.load(file)
        
        with open('.pkl/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        
        with open('.pkl/pca.pkl', 'rb') as file:
            pca = pickle.load(file)
        
        # Remove target column if it exists in expected_cols
        if "num" in expected_cols:
            expected_cols = [c for c in expected_cols if c != "num"]
        
        return model, expected_cols, scaler, pca
    
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load components
model, expected_cols, scaler, pca = load_model_components()

# ------------------------
# Page Header
# ------------------------
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This application uses machine learning to predict the risk of heart disease based on patient medical data.
Please enter the patient details below to get a prediction.""")

# Add disclaimer
st.info("⚠️ **Disclaimer**: This tool is for educational purposes only and should not replace professional medical advice.")

# ------------------------
# Create two columns for better layout
# ------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Patient Demographics")
    age = st.number_input("Age", min_value=1, max_value=120, value=50, help="Patient's age in years")
    sex = st.selectbox("Sex", ["male", "female"], help="Patient's biological sex")
    
    st.subheader("🩺 Vital Signs")
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=50, max_value=250, value=120, 
                              help="Resting blood pressure in mmHg")
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200,
                          help="Serum cholesterol level in mg/dl")
    thalch = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150,
                           help="Maximum heart rate achieved during stress test")

with col2:
    st.subheader("🫀 Heart-Related Tests")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                     help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], 
                      format_func=lambda x: "Yes" if x == 1 else "No",
                      help="Is fasting blood sugar > 120 mg/dl?")
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2],
                          help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")
    exang = st.selectbox("Exercise Induced Angina", [0, 1], 
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        help="Does exercise induce angina?")
    
    st.subheader("📊 Exercise Test Results")
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                             help="ST depression induced by exercise relative to rest")
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2],
                        help="0: Upsloping, 1: Flat, 2: Downsloping")
    ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0,
                        help="Number of major vessels colored by fluoroscopy")
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                       help="0: Normal, 1: Fixed defect, 2: Reversible defect, 3: Not described")

# ------------------------
# Prediction Section
# ------------------------
st.markdown("---")
st.subheader("🎯 Prediction")

if st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing patient data..."):
        try:
            # ------------------------
            # Prepare DataFrame
            # ------------------------
            input_df = pd.DataFrame({
                'age': [age],
                'trestbps': [trestbps],
                'chol': [chol],
                'thalch': [thalch],
                'oldpeak': [oldpeak],
                'ca': [ca],
                'sex': [sex],
                'cp': [cp],
                'fbs': [fbs],
                'restecg': [restecg],
                'exang': [exang],
                'slope': [slope],
                'thal': [thal]
            })

            # ------------------------
            # Encoding (same as training)
            # ------------------------
            input_df = pd.get_dummies(
                input_df,
                columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'],
                dtype=int
            )

            # Ensure all expected columns are present
            for col in expected_cols:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Reorder columns to match training data
            input_df = input_df[expected_cols]

            # ------------------------
            # Scaling 
            # ------------------------
            scale_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
            input_df[scale_cols] = scaler.transform(input_df[scale_cols])

            # ------------------------
            # PCA 
            # ------------------------
            X_pca = pca.transform(input_df)

            # ------------------------
            # Prediction
            # ------------------------
            prediction = model.predict(X_pca)[0]
            prediction_proba = model.predict_proba(X_pca)[0]
            
            # Display results
            if prediction == 1:
                st.error(f"⚠️ **HIGH RISK** of heart disease (Confidence: {prediction_proba[1]:.2%})")
                st.markdown("""
                **Recommendations:**
                - Consult with a cardiologist immediately
                - Consider lifestyle modifications (diet, exercise)
                - Regular monitoring of cardiovascular health
                """)
            else:
                st.success(f"✅ **LOW RISK** of heart disease (Confidence: {prediction_proba[0]:.2%})")
                st.markdown("""
                **Recommendations:**
                - Continue maintaining healthy lifestyle
                - Regular check-ups with healthcare provider
                - Monitor cardiovascular risk factors
                """)
            
            # Show probability breakdown
            st.subheader("📊 Prediction Confidence")
            prob_col1, prob_col2 = st.columns(2)
            
            with prob_col1:
                st.metric("Low Risk Probability", f"{prediction_proba[0]:.2%}")
            
            with prob_col2:
                st.metric("High Risk Probability", f"{prediction_proba[1]:.2%}")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please check your input values and try again.")

# ------------------------
# Additional Information
# ------------------------
with st.expander("ℹ️ About This Model"):
    st.markdown("""
    **Model Information:**
    - Algorithm: XGBoost Classifier
    - Features: 13 clinical parameters
    - Preprocessing: Standardization + PCA
    
    **Input Features:**
    - **Age**: Patient's age in years
    - **Sex**: Biological sex (male/female)
    - **Chest Pain Type**: Classification of chest pain (0-3)
    - **Resting Blood Pressure**: Blood pressure at rest (mmHg)
    - **Cholesterol**: Serum cholesterol level (mg/dl)
    - **Fasting Blood Sugar**: Whether fasting blood sugar > 120 mg/dl
    - **Resting ECG**: Electrocardiogram results at rest
    - **Max Heart Rate**: Maximum heart rate achieved during stress test
    - **Exercise Angina**: Whether exercise induces angina
    - **ST Depression**: ST depression induced by exercise
    - **ST Slope**: Slope of peak exercise ST segment
    - **Major Vessels**: Number of major vessels colored by fluoroscopy (0-3)
    - **Thalassemia**: Thalassemia test results
    
    **Note**: This model is trained on historical medical data and should be used as a screening tool only.
    Always consult with healthcare professionals for medical decisions.
    """)

# Footer
st.markdown("---")
st.header("*Nour Hatem*")
st.markdown("*Built with Streamlit • For educational purposes only*")
