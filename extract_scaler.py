import joblib
from pathlib import Path

def extract_scaler():
    scaler_path = Path("model_pipeline/models/scaler.pkl")
    if not scaler_path.exists():
        print("Scaler file not found!")
        return

    scaler = joblib.load(scaler_path)
    print("FEATURE_MEAN = [")
    for val in scaler.mean_:
        print(f"  {val},")
    print("];")
    
    print("\nFEATURE_SCALE = [")
    for val in scaler.scale_:
        print(f"  {val},")
    print("];")

if __name__ == "__main__":
    extract_scaler()
