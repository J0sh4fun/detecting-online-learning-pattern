from pathlib import Path
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import sys

def debug_export():
    ROOT_DIR = Path(__file__).resolve().parent
    model_pkl = ROOT_DIR / "model_pipeline" / "models" / "best_posture_model.pkl"
    out_path = ROOT_DIR / "model_pipeline" / "models" / "debug_posture.onnx"
    
    print(f"Loading model from: {model_pkl}")
    if not model_pkl.exists():
        print("Model file not found!")
        return

    model = joblib.load(model_pkl)
    print(f"Model loaded: {type(model)}")
    
    print("Converting to ONNX...")
    try:
        # We'll try without zipmap first as that was the goal
        options = {type(model): {"zipmap": False}}
        onx = convert_sklearn(
            model, 
            initial_types=[('float_input', FloatTensorType([None, 8]))],
            options=options
        )
        print("Conversion successful!")
        
        print(f"Saving to: {out_path}")
        out_path.write_bytes(onx.SerializeToString())
        print("Done!")
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_export()
