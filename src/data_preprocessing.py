import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.categorical_encoders = {}
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        return pd.read_csv(filepath)
    
    def preprocess(self, df, fit=False):
        """Preprocess the data"""
        df_copy = df.copy()
        
        # Handle categorical variables
        categorical_cols = ['education_level', 'employment_type']
        for col in categorical_cols:
            if col in df_copy.columns:
                if fit:
                    self.categorical_encoders[col] = LabelEncoder()
                    df_copy[col] = self.categorical_encoders[col].fit_transform(df_copy[col].astype(str))
                else:
                    df_copy[col] = self.categorical_encoders[col].transform(df_copy[col].astype(str))
        
        # Handle numerical features
        numerical_cols = ['age', 'income', 'credit_score', 'employment_years', 
                         'debt_ratio', 'loan_amount', 'interest_rate', 'dependents']
        
        # Extract target if present
        target = None
        if 'loan_approved' in df_copy.columns:
            target = df_copy['loan_approved'].values
            df_copy = df_copy.drop('loan_approved', axis=1)
        
        # Scale numerical features
        if fit:
            df_copy[numerical_cols] = self.scaler.fit_transform(df_copy[numerical_cols])
        else:
            df_copy[numerical_cols] = self.scaler.transform(df_copy[numerical_cols])
        
        return df_copy, target
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def save_scalers(self, filepath='models/scalers.joblib'):
        """Save scalers and encoders"""
        joblib.dump({
            'scaler': self.scaler,
            'encoders': self.categorical_encoders
        }, filepath)
    
    def load_scalers(self, filepath='models/scalers.joblib'):
        """Load scalers and encoders"""
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.categorical_encoders = data['encoders']