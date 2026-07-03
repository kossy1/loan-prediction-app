import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("LOAN APPROVAL PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # 1. Load and preprocess data
    print("\n[1/5] Loading data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('data/loan_data.csv')
    print(f"Loaded {len(df)} records")
    
    # 2. Preprocess
    print("\n[2/5] Preprocessing data...")
    X, y = preprocessor.preprocess(df, fit=True)
    X_train, X_val, y_train, y_val = preprocessor.split_data(X, y)
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # 3. Train models
    print("\n[3/5] Training models...")
    trainer = ModelTrainer()
    
    # Train Logistic Regression
    log_metrics = trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
    
    # Train XGBoost
    xgb_metrics = trainer.train_xgboost(X_train, y_train, X_val, y_val)
    
    # 4. Save models and preprocessor
    print("\n[4/5] Saving models...")
    trainer.save_models()
    preprocessor.save_scalers()
    
    # 5. Compare models
    print("\n[5/5] Model Comparison:")
    print("-" * 60)
    print(f"{'Metric':<15} {'Logistic':<15} {'XGBoost':<15} {'Difference':<10}")
    print("-" * 60)
    for metric in log_metrics.keys():
        diff = xgb_metrics[metric] - log_metrics[metric]
        print(f"{metric:<15} {log_metrics[metric]:<15.4f} {xgb_metrics[metric]:<15.4f} {diff:<10.4f}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Save best model info
    best_model = 'XGBoost' if xgb_metrics['roc_auc'] > log_metrics['roc_auc'] else 'Logistic Regression'
    print(f"\nBest Model: {best_model}")
    print(f"Best ROC-AUC: {max(log_metrics['roc_auc'], xgb_metrics['roc_auc']):.4f}")

if __name__ == "__main__":
    main()