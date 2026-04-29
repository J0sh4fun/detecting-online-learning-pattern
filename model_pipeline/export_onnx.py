import joblib
import warnings
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from ultralytics import YOLO

warnings.filterwarnings('ignore')

def main():
    print("Exporting Random Forest to ONNX...")
    # Load Scikit-Learn Model
    scaler = joblib.load('models/scaler.pkl')
    model = joblib.load('models/best_posture_model.pkl')
    
    # We have 9 input features
    initial_type = [('float_input', FloatTensorType([None, 9]))]
    
    # Convert Random Forest
    onx = convert_sklearn(model, initial_types=initial_type)
    
    with open("models/best_posture_model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    print("Saved RF to best_posture_model.onnx")
    
    # Export YOLO
    print("Exporting YOLOv8 to ONNX...")
    yolo = YOLO('yolov8s.pt')
    yolo.export(format='onnx', dynamic=False, opset=12) # ONNX web usually prefers standard opsets
    print("YOLO export complete.")

if __name__ == "__main__":
    main()
