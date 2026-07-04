import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

print("=" * 60)
print("TRAINING LOAN APPROVAL PREDICTION MODELS")
print("=" * 60)

# Generate synthetic data
np.random.seed(42)
n_samples = 5000

# Generate features
data = {
    'age': np.random.randint(18, 70, n_samples),
    'income': np.random.normal(50000, 20000, n_samples),
    'credit_score': np.random.normal(650, 100, n_samples),
    'employment_years': np.random.exponential(10, n_samples),
    'debt_ratio': np.random.uniform(0.1, 0.8, n_samples),
    'loan_amount': np.random.normal(100000, 50000, n_samples),
    'interest_rate': np.random.uniform(3, 15, n_samples),
    'dependents': np.random.randint(0, 5, n_samples),
    'education_level': np.random.randint(1, 6, n_samples),
    'employment_type': np.random.randint(1, 5, n_samples)
}

df = pd.DataFrame(data)

# Clean up values
df['income'] = np.abs(df['income'])
df['credit_score'] = np.clip(df['credit_score'], 300, 850).astype(int)
df['employment_years'] = np.clip(df['employment_years'], 0, 50).astype(int)
df['loan_amount'] = np.abs(df['loan_amount']).astype(int)
df['age'] = np.clip(df['age'], 18, 100)

# Generate target (loan approval)
approval_score = (
    (df['credit_score'] - 300) / 550 * 0.4 +
    (1 - df['debt_ratio']) * 0.3 +
    (df['income'] / 100000) * 0.2 +
    (df['employment_years'] / 20) * 0.1
)

# Add some randomness
approval_score += np.random.normal(0, 0.1, n_samples)
df['loan_approved'] = (approval_score > 0.5).astype(int)

print(f"\n📊 Dataset created with {len(df)} samples")
print(f"   Approved: {df['loan_approved'].sum()}")
print(f"   Rejected: {len(df) - df['loan_approved'].sum()}")

# Prepare features and target
feature_cols = ['age', 'income', 'credit_score', 'employment_years', 
                'debt_ratio', 'loan_amount', 'interest_rate', 
                'dependents', 'education_level', 'employment_type']

X = df[feature_cols]
y = df['loan_approved']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Data split:")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler with consistent naming
joblib.dump(scaler, 'models/scalers.joblib')
print("\n✅ Scaler saved to models/scalers.joblib")

# Train Logistic Regression
print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

logistic_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)
logistic_model.fit(X_train_scaled, y_train)

# Evaluate Logistic Regression
y_pred_log = logistic_model.predict(X_test_scaled)
log_accuracy = accuracy_score(y_test, y_pred_log)

print(f"✅ Logistic Regression trained!")
print(f"   Accuracy: {log_accuracy:.4f}")

# Train XGBoost
print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train_scaled, y_train)

# Evaluate XGBoost
y_pred_xgb = xgb_model.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)

print(f"✅ XGBoost trained!")
print(f"   Accuracy: {xgb_accuracy:.4f}")

# Save models
print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)

joblib.dump(logistic_model, 'models/logistic_model.pkl')
joblib.dump(xgb_model, 'models/xgboost_model.pkl')

print("✅ Logistic Regression saved to: models/logistic_model.pkl")
print("✅ XGBoost saved to: models/xgboost_model.pkl")
print("✅ Scaler saved to: models/scalers.joblib")

# Check file sizes
print("\n📦 Model file sizes:")
for model_file in ['logistic_model.pkl', 'xgboost_model.pkl', 'scalers.joblib']:
    file_path = f'models/{model_file}'
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / (1024 * 1024)
        print(f"   {model_file}: {size:.2f} MB")
    else:
        print(f"   {model_file}: NOT FOUND")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)