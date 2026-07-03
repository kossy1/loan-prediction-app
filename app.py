import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# Title and description
st.title("🏦 Loan Approval Prediction System")
st.markdown("""
This application uses machine learning to predict loan approval status.
Enter the applicant details below to get a prediction.
""")

# Sidebar for API configuration
st.sidebar.header("⚙️ Configuration")
api_url = st.sidebar.text_input(
    "API URL",
    value="http://localhost:8000" if st.sidebar.checkbox("Local Development") else "https://your-app.vercel.app"
)
model_type = st.sidebar.selectbox(
    "Model Type",
    ["ensemble", "logistic", "xgboost"],
    help="Ensemble combines both models for better accuracy"
)

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Information")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    income = st.number_input("Annual Income ($)", min_value=0, max_value=1000000, value=60000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720)
    employment_years = st.number_input("Years of Employment", min_value=0.0, max_value=50.0, value=8.0)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)

with col2:
    st.subheader("💰 Financial Information")
    debt_ratio = st.slider("Debt to Income Ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, max_value=1000000, value=150000)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=4.5, step=0.5)
    
    st.subheader("📋 Additional Information")
    education_level = st.selectbox(
        "Education Level",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: ["High School", "Some College", "Bachelor's", "Master's", "PhD"][x-1],
        index=3
    )
    employment_type = st.selectbox(
        "Employment Type",
        options=[1, 2, 3, 4],
        format_func=lambda x: ["Self-Employed", "Full-Time", "Part-Time", "Contract"][x-1],
        index=1
    )

# Prepare input data
input_data = {
    "age": age,
    "income": income,
    "credit_score": credit_score,
    "employment_years": employment_years,
    "debt_ratio": debt_ratio,
    "loan_amount": loan_amount,
    "interest_rate": interest_rate,
    "dependents": dependents,
    "education_level": education_level,
    "employment_type": employment_type
}

# Prediction button
if st.button("🔮 Predict Loan Approval", type="primary", use_container_width=True):
    try:
        # Show loading spinner
        with st.spinner("Making prediction..."):
            # Send request to API
            response = requests.post(
                f"{api_url}/predict",
                json=input_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                # Create metrics display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Status",
                        result['status'],
                        delta="Approved ✅" if result['prediction'] == 1 else "Rejected ❌"
                    )
                
                with col2:
                    st.metric(
                        "Confidence",
                        f"{result['confidence']*100:.1f}%",
                        delta=f"Probability: {result['probability']*100:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Model Used",
                        result['model_used'].upper(),
                        delta=f"ID: {result['prediction']}"
                    )
                
                # Show detailed results
                with st.expander("📊 Detailed Results", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json({
                            "Prediction": "Approved" if result['prediction'] == 1 else "Rejected",
                            "Probability": f"{result['probability']*100:.2f}%",
                            "Confidence": f"{result['confidence']*100:.2f}%",
                            "Model": result['model_used'],
                            "Timestamp": result['timestamp']
                        })
                    with col2:
                        # Create gauge chart for probability
                        prob = result['probability'] * 100
                        st.markdown(f"""
                        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                            <h4>Approval Probability</h4>
                            <div style="background-color: #e0e0e0; height: 30px; border-radius: 15px; overflow: hidden;">
                                <div style="background-color: {'#4CAF50' if prob > 50 else '#f44336'}; 
                                            width: {prob}%; height: 100%; 
                                            display: flex; align-items: center; justify-content: center;
                                            color: white; font-weight: bold; transition: width 1s;">
                                    {prob:.1f}%
                                </div>
                            </div>
                            <p style="margin-top: 10px;">
                                Threshold: <b>50%</b> • 
                                Current: <b style="color: {'#4CAF50' if prob > 50 else '#f44336'}">
                                    {prob:.1f}%
                                </b>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show input summary
                with st.expander("📝 Input Summary"):
                    df = pd.DataFrame([input_data])
                    st.dataframe(df, use_container_width=True)
                    
            else:
                st.error(f"❌ Error: {response.status_code} - {response.text}")
                
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the API. Make sure the server is running.")
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. Please try again.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

# Additional information in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    st.markdown("""
    - **Logistic Regression**: Linear model with regularization
    - **XGBoost**: Gradient boosting with high accuracy
    - **Ensemble**: Combined predictions for better results
    """)
    
    st.markdown("---")
    st.markdown("### 📋 Feature Descriptions")
    st.markdown("""
    - **Age**: Applicant's age
    - **Income**: Annual income in USD
    - **Credit Score**: 300-850 range
    - **Employment Years**: Years of employment
    - **Debt Ratio**: Debt to income ratio
    - **Loan Amount**: Requested amount
    - **Interest Rate**: Interest rate in %
    - **Dependents**: Number of dependents
    - **Education Level**: 1-5 scale
    - **Employment Type**: 1-4 categories
    """)
    
    st.markdown("---")
    st.markdown("### 🚀 Deployment")
    st.markdown("""
    Deployed on:
    - **Vercel**: Serverless API
    - **Streamlit**: Web Interface
    - **FastAPI**: Backend API
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    Made with ❤️ using Streamlit, FastAPI, and Machine Learning
</div>
""", unsafe_allow_html=True)