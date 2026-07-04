import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
import warnings
import numpy as np

warnings.filterwarnings('ignore')

app = FastAPI(
    title="Loan Approval Prediction API",
    description="Predict loan approval using Logistic Regression",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
scaler = None
models_loaded = False

# Request model
class LoanApplication(BaseModel):
    age: int = Field(ge=18, le=100)
    income: float = Field(ge=0, le=1000000)
    credit_score: int = Field(ge=300, le=850)
    employment_years: float = Field(ge=0, le=50)
    debt_ratio: float = Field(ge=0, le=1)
    loan_amount: float = Field(ge=0, le=1000000)
    interest_rate: float = Field(ge=0, le=30)
    dependents: int = Field(ge=0, le=10)
    education_level: int = Field(ge=1, le=5)
    employment_type: int = Field(ge=1, le=4)

# Response model
class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    status: str
    confidence: float
    model_used: str
    timestamp: str

def load_model_gracefully():
    """Try to load model, return success status"""
    global model, scaler, models_loaded
    
    try:
        import joblib
        
        # Check if files exist
        if not os.path.exists('models/logistic_model.pkl'):
            print("⚠️ Model file not found")
            models_loaded = False
            return False
        
        if not os.path.exists('models/scalers.joblib'):
            print("⚠️ Scaler file not found")
            models_loaded = False
            return False
        
        # Load model and scaler
        model = joblib.load('models/logistic_model.pkl')
        scaler = joblib.load('models/scalers.joblib')
        
        models_loaded = True
        print("✅ Logistic Regression model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        models_loaded = False
        return False

def predict_with_model(data):
    """Predict using Logistic Regression"""
    global model, scaler
    
    if not models_loaded:
        return None, None
    
    try:
        features = np.array([[
            data['age'],
            data['income'],
            data['credit_score'],
            data['employment_years'],
            data['debt_ratio'],
            data['loan_amount'],
            data['interest_rate'],
            data['dependents'],
            data['education_level'],
            data['employment_type']
        ]])
        
        features_scaled = scaler.transform(features)
        probability = model.predict_proba(features_scaled)[0][1]
        prediction = 1 if probability >= 0.5 else 0
        
        return prediction, probability
        
    except Exception as e:
        print(f"⚠️ Error predicting: {e}")
        return None, None

def predict_with_fallback(data):
    """Rule-based fallback prediction"""
    score = 0
    if data['credit_score'] > 650: score += 1
    if data['debt_ratio'] < 0.4: score += 1
    if data['employment_years'] > 5: score += 1
    if data['income'] > 50000: score += 1
    if data['loan_amount'] < 200000: score += 1
    
    prediction = 1 if score >= 3 else 0
    probability = 0.6 + (score * 0.08)
    probability = min(probability, 0.95)
    
    return prediction, probability

# Load model at startup
try:
    load_model_gracefully()
except Exception as e:
    print(f"⚠️ Startup error: {e}")
    models_loaded = False

@app.get("/")
async def root():
    return {
        "message": "Loan Approval Prediction API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": models_loaded,
        "model_type": "Logistic Regression"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    try:
        data = application.dict()
        
        # Try ML model first
        prediction, probability = predict_with_model(data)
        
        # Fallback to rule-based if needed
        if prediction is None:
            prediction, probability = predict_with_fallback(data)
            model_used = "rule-based (fallback)"
        else:
            model_used = "Logistic Regression"
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "status": "Approved" if prediction == 1 else "Rejected",
            "confidence": float(probability if prediction == 1 else 1 - probability),
            "model_used": model_used,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")