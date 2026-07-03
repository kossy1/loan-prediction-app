# generate_data.py
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

np.random.seed(42)

# Create synthetic loan data
n_samples = 10000
n_features = 10

# Generate features
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=8,
    n_redundant=2,
    n_clusters_per_class=1,
    flip_y=0.05,
    random_state=42
)

# Create DataFrame
feature_names = ['age', 'income', 'credit_score', 'employment_years', 
                 'debt_ratio', 'loan_amount', 'interest_rate', 
                 'dependents', 'education_level', 'employment_type']
df = pd.DataFrame(X, columns=feature_names)
df['loan_approved'] = y

# Transform some columns to make them more realistic
df['age'] = np.abs(df['age'] * 20 + 30).astype(int)  # 30-70 years
df['income'] = np.abs(df['income'] * 50000 + 30000).astype(int)  # 30k-80k
df['credit_score'] = np.clip(df['credit_score'] * 100 + 650, 300, 850).astype(int)
df['employment_years'] = np.abs(df['employment_years'] * 10 + 2).astype(int)
df['debt_ratio'] = np.clip(df['debt_ratio'] * 0.3 + 0.3, 0.1, 0.8)
df['loan_amount'] = np.abs(df['loan_amount'] * 50000 + 10000).astype(int)
df['interest_rate'] = np.clip(df['interest_rate'] * 5 + 5, 3, 20)
df['dependents'] = np.clip(df['dependents'] * 2 + 1, 0, 6).astype(int)
df['education_level'] = np.clip(df['education_level'] * 2 + 1, 1, 5).astype(int)
df['employment_type'] = np.clip(df['employment_type'] * 2 + 1, 1, 4).astype(int)

# Save to CSV
df.to_csv('data/loan_data.csv', index=False)
print("Sample data created successfully!")