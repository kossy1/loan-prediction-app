import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import your modules
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.utils import validate_input, preprocess_input, format_prediction_response

# Initialize FastAPI app
app = FastAPI(
    title="Loan Approval Prediction API",
    description="Machine Learning API for predicting loan approval using Logistic Regression and XGBoost",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
preprocessor = DataPreprocessor()
trainer = ModelTrainer()

# Define request model - Pydantic v2 compatible
class LoanApplication(BaseModel):
    age: int = Field(ge=18, le=100, description="Applicant's age")
    income: float = Field(ge=0, le=1000000, description="Annual income")
    credit_score: int = Field(ge=300, le=850, description="Credit score")
    employment_years: float = Field(ge=0, le=50, description="Years of employment")
    debt_ratio: float = Field(ge=0, le=1, description="Debt to income ratio")
    loan_amount: float = Field(ge=0, le=1000000, description="Requested loan amount")
    interest_rate: float = Field(ge=0, le=30, description="Interest rate")
    dependents: int = Field(ge=0, le=10, description="Number of dependents")
    education_level: int = Field(ge=1, le=5, description="Education level (1-5)")
    employment_type: int = Field(ge=1, le=4, description="Employment type (1-4)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "income": 60000,
                "credit_score": 720,
                "employment_years": 8,
                "debt_ratio": 0.35,
                "loan_amount": 150000,
                "interest_rate": 4.5,
                "dependents": 2,
                "education_level": 4,
                "employment_type": 2
            }
        }

# Define response model
class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    status: str
    confidence: float
    model_used: str
    timestamp: str

# Load models at startup
@app.on_event("startup")
async def load_models():
    """Load trained models and preprocessor"""
    try:
        # Try to load saved models
        preprocessor.load_scalers('models/scalers.joblib')
        trainer.load_models('models/logistic_model.pkl', 'models/xgboost_model.pkl')
        print("✅ Models and preprocessor loaded successfully!")
    except Exception as e:
        print(f"⚠️  Error loading models: {e}")
        print("Creating dummy models for deployment...")
        
        # Create dummy models for deployment if not available
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler
        
        trainer.logistic_model = LogisticRegression()
        trainer.xgboost_model = XGBClassifier()
        preprocessor.scaler = StandardScaler()
        print("✅ Dummy models created for deployment")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Loan Approval Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/": "GET - This information",
            "/health": "GET - Health check",
            "/predict": "POST - Predict using ensemble model",
            "/predict/logistic": "POST - Predict using Logistic Regression",
            "/predict/xgboost": "POST - Predict using XGBoost"
        },
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": trainer.logistic_model is not None and trainer.xgboost_model is not None,
        "preprocessor_loaded": preprocessor.scaler is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    """Predict loan approval using ensemble of Logistic Regression and XGBoost"""
    try:
        # Validate input
        validate_input(application.model_dump())
        
        # Preprocess
        X = preprocess_input(application.model_dump(), preprocessor)
        
        # Predict using ensemble
        prediction, probability = trainer.predict_loan_approval(X, model_type='ensemble')
        
        # Format response
        response = format_prediction_response(prediction, probability, 'ensemble')
        return response
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict/logistic", response_model=PredictionResponse)
async def predict_logistic(application: LoanApplication):
    """Predict loan approval using Logistic Regression only"""
    try:
        validate_input(application.model_dump())
        X = preprocess_input(application.model_dump(), preprocessor)
        prediction, probability = trainer.predict_loan_approval(X, model_type='logistic')
        response = format_prediction_response(prediction, probability, 'logistic')
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/xgboost", response_model=PredictionResponse)
async def predict_xgboost(application: LoanApplication):
    """Predict loan approval using XGBoost only"""
    try:
        validate_input(application.model_dump())
        X = preprocess_input(application.model_dump(), preprocessor)
        prediction, probability = trainer.predict_loan_approval(X, model_type='xgboost')
        response = format_prediction_response(prediction, probability, 'xgboost')
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")