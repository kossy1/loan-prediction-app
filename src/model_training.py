import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.logistic_model = None
        self.xgboost_model = None
        self.best_logistic_params = None
        self.best_xgboost_params = None
        
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """Train Logistic Regression model with hyperparameter tuning"""
        print("Training Logistic Regression...")
        
        # Hyperparameter grid
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000],
            'class_weight': ['balanced', None]
        }
        
        # Grid search
        logistic = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(
            logistic, param_grid, cv=5, scoring='roc_auc', 
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        self.best_logistic_params = grid_search.best_params_
        self.logistic_model = grid_search.best_estimator_
        
        # Evaluate
        y_pred = self.logistic_model.predict(X_val)
        y_prob = self.logistic_model.predict_proba(X_val)[:, 1]
        
        metrics = self._calculate_metrics(y_val, y_pred, y_prob)
        print(f"Logistic Regression Metrics:")
        self._print_metrics(metrics)
        
        return metrics
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model with hyperparameter tuning"""
        print("\nTraining XGBoost...")
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        
        # Grid search (reduced for faster training)
        xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        grid_search = GridSearchCV(
            xgb, param_grid, cv=3, scoring='roc_auc', 
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        self.best_xgboost_params = grid_search.best_params_
        self.xgboost_model = grid_search.best_estimator_
        
        # Evaluate
        y_pred = self.xgboost_model.predict(X_val)
        y_prob = self.xgboost_model.predict_proba(X_val)[:, 1]
        
        metrics = self._calculate_metrics(y_val, y_pred, y_prob)
        print(f"XGBoost Metrics:")
        self._print_metrics(metrics)
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate various metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob)
        }
    
    def _print_metrics(self, metrics):
        """Print metrics in a formatted way"""
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
    
    def save_models(self, logistic_path='models/logistic_model.pkl', 
                   xgboost_path='models/xgboost_model.pkl'):
        """Save trained models"""
        joblib.dump(self.logistic_model, logistic_path)
        joblib.dump(self.xgboost_model, xgboost_path)
        print(f"\nModels saved successfully!")
    
    def load_models(self, logistic_path='models/logistic_model.pkl',
                   xgboost_path='models/xgboost_model.pkl'):
        """Load trained models"""
        self.logistic_model = joblib.load(logistic_path)
        self.xgboost_model = joblib.load(xgboost_path)
        print("Models loaded successfully!")
    
    def predict_loan_approval(self, features, model_type='ensemble'):
        """Predict loan approval for new features"""
        if model_type == 'logistic':
            if self.logistic_model is None:
                raise ValueError("Logistic Regression model not loaded!")
            prob = self.logistic_model.predict_proba(features)[:, 1]
            pred = self.logistic_model.predict(features)
        elif model_type == 'xgboost':
            if self.xgboost_model is None:
                raise ValueError("XGBoost model not loaded!")
            prob = self.xgboost_model.predict_proba(features)[:, 1]
            pred = self.xgboost_model.predict(features)
        elif model_type == 'ensemble':
            # Use both models and average predictions
            if self.logistic_model is None or self.xgboost_model is None:
                raise ValueError("Both models must be loaded for ensemble!")
            
            prob_log = self.logistic_model.predict_proba(features)[:, 1]
            prob_xgb = self.xgboost_model.predict_proba(features)[:, 1]
            prob = (prob_log + prob_xgb) / 2
            pred = (prob >= 0.5).astype(int)
        else:
            raise ValueError("Invalid model type!")
        
        return pred, prob