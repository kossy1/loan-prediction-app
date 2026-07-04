import requests
import json

# Your deployed API URL
url = "https://loan-approval-pnim15q48-kossyvibes-7750s-projects.vercel.app/predict"

# Test cases
test_cases = [
    {
        "name": "Good Candidate (Should Approve)",
        "data": {
            "age": 35,
            "income": 75000,
            "credit_score": 780,
            "employment_years": 10,
            "debt_ratio": 0.25,
            "loan_amount": 120000,
            "interest_rate": 3.5,
            "dependents": 2,
            "education_level": 5,
            "employment_type": 2
        }
    },
    {
        "name": "Bad Candidate (Should Reject)",
        "data": {
            "age": 22,
            "income": 25000,
            "credit_score": 580,
            "employment_years": 1,
            "debt_ratio": 0.75,
            "loan_amount": 300000,
            "interest_rate": 12.5,
            "dependents": 3,
            "education_level": 2,
            "employment_type": 3
        }
    },
    {
        "name": "Borderline Candidate",
        "data": {
            "age": 30,
            "income": 45000,
            "credit_score": 650,
            "employment_years": 4,
            "debt_ratio": 0.45,
            "loan_amount": 180000,
            "interest_rate": 6.5,
            "dependents": 1,
            "education_level": 3,
            "employment_type": 1
        }
    }
]

print("=" * 60)
print("TESTING LOAN APPROVAL API")
print("=" * 60)

for test in test_cases:
    print(f"\n📊 Testing: {test['name']}")
    print("-" * 40)
    
    try:
        response = requests.post(url, json=test['data'])
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Status: {result['status']}")
            print(f"   📈 Probability: {result['probability']:.2%}")
            print(f"   🎯 Confidence: {result['confidence']:.2%}")
            print(f"   🤖 Model: {result['model_used']}")
        else:
            print(f"   ❌ Error: {response.status_code}")
            print(f"   {response.text}")
            
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
    
    print("-" * 40)

print("\n✅ Testing complete!")