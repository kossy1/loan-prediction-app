import pandas as pd
import numpy as np
import json
from datetime import datetime

def validate_input(data):
    """Validate input data"""
    required_fields = ['age', 'income', 'credit_score', 'employment_years', 
                      'debt_ratio', 'loan_amount', 'interest_rate', 
                      'dependents', 'education_level', 'employment_type']
    
    # Check for missing fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Missing fields: {missing_fields}")
    
    # Validate ranges
    validations = {
        'age': (18, 100),
        'income': (0, 1000000),
        'credit_score': (300, 850),
        'employment_years': (0, 50),
        'debt_ratio': (0, 1),
        'loan_amount': (0, 1000000),
        'interest_rate': (0, 30),
        'dependents': (0, 10),
        'education_level': (1, 5),
        'employment_type': (1, 4)
    }
    
    for field, (min_val, max_val) in validations.items():
        if field in data:
            value = data[field]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{field} must be between {min_val} and {max_val}")
    
    return True

def preprocess_input(data, preprocessor):
    """Preprocess input data for prediction"""
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Preprocess
    X, _ = preprocessor.preprocess(df, fit=False)
    
    return X

def format_prediction_response(prediction, probability, model_type='ensemble'):
    """Format prediction response"""
    return {
        'prediction': int(prediction[0]),
        'probability': float(probability[0]),
        'status': 'Approved' if prediction[0] == 1 else 'Rejected',
        'confidence': float(probability[0]) if prediction[0] == 1 else 1 - float(probability[0]),
        'model_used': model_type,
        'timestamp': datetime.now().isoformat()
    }

def get_feature_importance(model, feature_names):
    """Get feature importance from model"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None
    
    # Sort features by importance
    feature_importance = sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True
    )
    
    return feature_importance