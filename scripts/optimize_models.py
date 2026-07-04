import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def optimize_model():
    # Load models
    logistic = joblib.load('models/logistic_model.pkl')
    xgboost_model = joblib.load('models/xgboost_model.pkl')
    
    # Reduce precision
    if hasattr(logistic, 'coef_'):
        logistic.coef_ = logistic.coef_.astype(np.float32)
    if hasattr(logistic, 'intercept_'):
        logistic.intercept_ = logistic.intercept_.astype(np.float32)
    
    # Save compressed models
    joblib.dump(logistic, 'models/logistic_model_compressed.pkl', compress=3)
    joblib.dump(xgboost_model, 'models/xgboost_model_compressed.pkl', compress=3)
    
    print("Models compressed successfully!")

if __name__ == "__main__":
    optimize_model()