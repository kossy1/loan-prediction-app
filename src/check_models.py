import os
import joblib
import glob

print("=" * 60)
print("CHECKING MODEL FILES")
print("=" * 60)

# Check if models directory exists
models_dir = 'models'
if not os.path.exists(models_dir):
    print(f"❌ Models directory not found: {models_dir}")
    print("Please run: python train_models.py")
    exit()

# List all files in models directory
print(f"\n📁 Files in {models_dir}:")
for file in os.listdir(models_dir):
    file_path = os.path.join(models_dir, file)
    size = os.path.getsize(file_path) / (1024 * 1024)
    print(f"   {file}: {size:.2f} MB")

# Check specific files
required_files = ['logistic_model.pkl', 'xgboost_model.pkl', 'scalers.joblib']
print("\n🔍 Checking required files:")
for file in required_files:
    file_path = os.path.join(models_dir, file)
    if os.path.exists(file_path):
        print(f"   ✅ {file} - FOUND")
        try:
            # Try to load the file
            data = joblib.load(file_path)
            if file == 'logistic_model.pkl':
                print(f"      Type: {type(data).__name__}")
            elif file == 'xgboost_model.pkl':
                print(f"      Type: {type(data).__name__}")
            elif file == 'scalers.joblib':
                if isinstance(data, dict):
                    print(f"      Contains: {list(data.keys())}")
                else:
                    print(f"      Type: {type(data).__name__}")
        except Exception as e:
            print(f"      ⚠️ Error loading: {e}")
    else:
        print(f"   ❌ {file} - MISSING")

print("\n" + "=" * 60)