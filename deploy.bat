@echo off
echo Creating deployment package...

REM Ensure models are in the right place
if not exist "models\logistic_model.pkl" (
    echo Error: logistic_model.pkl not found!
    echo Please run: python train_models.py first
    exit /b 1
)

if not exist "models\xgboost_model.pkl" (
    echo Error: xgboost_model.pkl not found!
    echo Please run: python train_models.py first
    exit /b 1
)

if not exist "models\scalers.joblib" (
    echo Error: scalers.joblib not found!
    echo Please run: python train_models.py first
    exit /b 1
)

echo All models found! Deploying to Vercel...
vercel --prod --force