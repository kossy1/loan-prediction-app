from setuptools import setup, find_packages

setup(
    name="loan-approval-prediction",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scikit-learn==1.3.2",
        "xgboost==1.7.6",
        "joblib==1.3.2",
        "fastapi==0.109.0",
        "uvicorn==0.27.0",
        "pydantic==2.5.3",
        "starlette==0.35.1",
        "streamlit==1.29.0",
        "python-dotenv==1.0.0",
        "python-multipart==0.0.6",
        "httpx==0.26.0",
    ],
    python_requires=">=3.10",
)