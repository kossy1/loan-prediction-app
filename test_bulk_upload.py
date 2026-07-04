import requests
import json
import pandas as pd
from io import StringIO

# Base URL (replace with your actual deployment URL)
BASE_URL = "https://loan-approval-9xlp3beaz-kossyvibes-7750s-projects.vercel.app"
# For local testing
# BASE_URL = "http://localhost:8000"

def test_single_prediction():
    """Test single prediction"""
    print("=" * 60)
    print("TESTING SINGLE PREDICTION")
    print("=" * 60)
    
    data = {
        "age": 35,
        "income": 60000,
        "credit_score": 720,
        "employment_years": 8,
        "debt_ratio": 0.35,
        "loan_amount": 150000,
        "interest_rate": 4.5,
        "dependents": 2,
        "education_level": 4,
        "employment_type": 2
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_bulk_json():
    """Test bulk JSON prediction"""
    print("\n" + "=" * 60)
    print("TESTING BULK JSON PREDICTION")
    print("=" * 60)
    
    data = [
        {
            "age": 35,
            "income": 60000,
            "credit_score": 720,
            "employment_years": 8,
            "debt_ratio": 0.35,
            "loan_amount": 150000,
            "interest_rate": 4.5,
            "dependents": 2,
            "education_level": 4,
            "employment_type": 2
        },
        {
            "age": 42,
            "income": 85000,
            "credit_score": 780,
            "employment_years": 12,
            "debt_ratio": 0.25,
            "loan_amount": 200000,
            "interest_rate": 3.8,
            "dependents": 3,
            "education_level": 5,
            "employment_type": 2
        },
        {
            "age": 23,
            "income": 25000,
            "credit_score": 580,
            "employment_years": 1,
            "debt_ratio": 0.75,
            "loan_amount": 50000,
            "interest_rate": 12.5,
            "dependents": 0,
            "education_level": 2,
            "employment_type": 3
        }
    ]
    
    response = requests.post(f"{BASE_URL}/predict/bulk/json", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Total: {result['total']}")
    print(f"Approved: {result['approved']}")
    print(f"Rejected: {result['rejected']}")
    print(f"Summary: {json.dumps(result['summary'], indent=2)}")

def test_bulk_csv():
    """Test bulk CSV upload"""
    print("\n" + "=" * 60)
    print("TESTING BULK CSV UPLOAD")
    print("=" * 60)
    
    # Create CSV data
    csv_data = """age,income,credit_score,employment_years,debt_ratio,loan_amount,interest_rate,dependents,education_level,employment_type
35,60000,720,8,0.35,150000,4.5,2,4,2
42,85000,780,12,0.25,200000,3.8,3,5,2
28,35000,620,3,0.55,100000,7.2,1,3,1
55,95000,800,20,0.15,300000,4.0,2,5,2
23,25000,580,1,0.75,50000,12.5,0,2,3"""
    
    files = {
        'file': ('test_loans.csv', csv_data, 'text/csv')
    }
    
    response = requests.post(f"{BASE_URL}/predict/bulk", files=files)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Total: {result['total']}")
    print(f"Approved: {result['approved']}")
    print(f"Rejected: {result['rejected']}")
    print(f"Summary: {json.dumps(result['summary'], indent=2)}")
    
    # Show first few results
    print("\nFirst 3 results:")
    for r in result['results'][:3]:
        print(f"  Row {r['row']}: {r['status']} (Prob: {r['probability']:.2%})")

if __name__ == "__main__":
    print("Starting API Tests...")
    print(f"Base URL: {BASE_URL}")
    
    try:
        test_single_prediction()
        test_bulk_json()
        test_bulk_csv()
    except Exception as e:
        print(f"Error: {e}")