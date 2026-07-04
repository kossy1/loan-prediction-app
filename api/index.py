from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np
import os
import warnings
import joblib
import pandas as pd
import io
import json

warnings.filterwarnings('ignore')

app = FastAPI(
    title="Loan Approval Prediction API",
    version="1.0.0",
    description="Predict loan approval with bulk CSV upload support"
)

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

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    status: str
    confidence: float
    model_used: str
    timestamp: str

class BulkPredictionResponse(BaseModel):
    total: int
    approved: int
    rejected: int
    results: list
    summary: dict
    timestamp: str

def load_models():
    """Try to load ML models"""
    global model, scaler, models_loaded
    
    try:
        if not os.path.exists('models'):
            print("📁 Models directory not found")
            models_loaded = False
            return
        
        model_path = 'models/logistic_model.pkl'
        scaler_path = 'models/scalers.joblib'
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"⚠️ Model files not found")
            models_loaded = False
            return
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        models_loaded = True
        print("✅ Models loaded successfully!")
        
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
        models_loaded = False

def predict_ml(data):
    """Predict using ML model"""
    global model, scaler
    
    try:
        features = np.array([[
            data['age'], data['income'], data['credit_score'],
            data['employment_years'], data['debt_ratio'], data['loan_amount'],
            data['interest_rate'], data['dependents'], data['education_level'],
            data['employment_type']
        ]])
        
        features_scaled = scaler.transform(features)
        probability = model.predict_proba(features_scaled)[0][1]
        prediction = 1 if probability >= 0.5 else 0
        
        return prediction, probability
        
    except Exception as e:
        print(f"⚠️ ML prediction error: {e}")
        return None, None

def predict_rule_based(data):
    """Rule-based prediction"""
    score = 0
    
    if data['credit_score'] > 650:
        score += 1
    if data['debt_ratio'] < 0.4:
        score += 1
    if data['employment_years'] > 5:
        score += 1
    if data['income'] > 50000:
        score += 1
    if data['loan_amount'] < 200000:
        score += 1
    if data['dependents'] <= 2:
        score += 1
    
    prediction = 1 if score >= 4 else 0
    probability = 0.5 + (score * 0.08)
    probability = min(probability, 0.95)
    
    return prediction, probability

def predict_single(data):
    """Make single prediction"""
    if models_loaded:
        prediction, probability = predict_ml(data)
        if prediction is not None:
            model_used = "Logistic Regression"
        else:
            prediction, probability = predict_rule_based(data)
            model_used = "rule-based (fallback)"
    else:
        prediction, probability = predict_rule_based(data)
        model_used = "rule-based"
    
    return prediction, probability, model_used

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/")
async def root():
    return {
        "message": "Loan Approval Prediction API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": models_loaded,
        "endpoints": {
            "/": "GET - API info",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation",
            "/predict": "POST - Predict single loan",
            "/predict/bulk": "POST - Predict bulk loans from CSV",
            "/predict/bulk/json": "POST - Predict bulk loans from JSON"
        }
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
    """Predict single loan approval"""
    try:
        data = application.dict()
        prediction, probability, model_used = predict_single(data)
        
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/bulk", response_model=BulkPredictionResponse)
async def predict_bulk_csv(file: UploadFile = File(...)):
    """Predict multiple loans from CSV file"""
    try:
        # Validate file type
        if not file.filename.endswith(('.csv', '.txt')):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file format. Please upload a CSV file."
            )
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_columns = [
            'age', 'income', 'credit_score', 'employment_years',
            'debt_ratio', 'loan_amount', 'interest_rate',
            'dependents', 'education_level', 'employment_type'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Process each row
        results = []
        approved_count = 0
        rejected_count = 0
        
        for index, row in df.iterrows():
            try:
                # Convert row to dict
                data = row.to_dict()
                
                # Validate ranges
                if not (18 <= data['age'] <= 100):
                    data['age'] = max(18, min(100, data['age']))
                if not (300 <= data['credit_score'] <= 850):
                    data['credit_score'] = max(300, min(850, data['credit_score']))
                
                # Predict
                prediction, probability, model_used = predict_single(data)
                
                result = {
                    "row": int(index + 1),
                    "prediction": int(prediction),
                    "probability": float(probability),
                    "status": "Approved" if prediction == 1 else "Rejected",
                    "confidence": float(probability if prediction == 1 else 1 - probability),
                    "model_used": model_used,
                    "input_data": data
                }
                results.append(result)
                
                if prediction == 1:
                    approved_count += 1
                else:
                    rejected_count += 1
                    
            except Exception as e:
                results.append({
                    "row": int(index + 1),
                    "error": str(e),
                    "status": "Error"
                })
                rejected_count += 1
        
        # Generate summary statistics
        total = len(results)
        
        # Calculate average probability for approved loans
        approved_probs = [r['probability'] for r in results if r.get('status') == 'Approved']
        avg_prob = sum(approved_probs) / len(approved_probs) if approved_probs else 0
        
        summary = {
            "total_loans": total,
            "approved": approved_count,
            "rejected": rejected_count,
            "approval_rate": f"{(approved_count/total*100):.1f}%",
            "average_probability_approved": f"{avg_prob:.2%}",
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "total": total,
            "approved": approved_count,
            "rejected": rejected_count,
            "results": results,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Bulk prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/bulk/json")
async def predict_bulk_json(applications: list[LoanApplication]):
    """Predict multiple loans from JSON array"""
    try:
        results = []
        approved_count = 0
        rejected_count = 0
        
        for i, app_data in enumerate(applications):
            try:
                data = app_data.dict()
                prediction, probability, model_used = predict_single(data)
                
                result = {
                    "id": i + 1,
                    "prediction": int(prediction),
                    "probability": float(probability),
                    "status": "Approved" if prediction == 1 else "Rejected",
                    "confidence": float(probability if prediction == 1 else 1 - probability),
                    "model_used": model_used
                }
                results.append(result)
                
                if prediction == 1:
                    approved_count += 1
                else:
                    rejected_count += 1
                    
            except Exception as e:
                results.append({
                    "id": i + 1,
                    "error": str(e),
                    "status": "Error"
                })
                rejected_count += 1
        
        total = len(results)
        
        return {
            "total": total,
            "approved": approved_count,
            "rejected": rejected_count,
            "results": results,
            "summary": {
                "total_loans": total,
                "approved": approved_count,
                "rejected": rejected_count,
                "approval_rate": f"{(approved_count/total*100):.1f}%",
                "timestamp": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ JSON bulk prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {
        "error": str(exc),
        "message": "An error occurred processing your request"
    }