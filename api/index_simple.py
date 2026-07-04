# api/index_simple.py - No ML dependencies
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

class LoanApplication(BaseModel):
    age: int
    income: float
    credit_score: int
    employment_years: float
    debt_ratio: float
    loan_amount: float
    interest_rate: float
    dependents: int
    education_level: int
    employment_type: int

@app.post("/predict")
async def predict(data: LoanApplication):
    data = data.dict()
    
    # Rule-based prediction
    score = 0
    if data['credit_score'] > 650: score += 1
    if data['debt_ratio'] < 0.4: score += 1
    if data['employment_years'] > 5: score += 1
    if data['income'] > 50000: score += 1
    
    prediction = 1 if score >= 3 else 0
    probability = 0.6 + (score * 0.08)
    
    return {
        "prediction": prediction,
        "probability": min(probability, 0.95),
        "status": "Approved" if prediction == 1 else "Rejected",
        "confidence": min(probability, 0.95),
        "model_used": "rule-based",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}