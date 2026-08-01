import sys
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.utils.helpers import get_base_path

# Load the YOLOv8 model
model_path = get_base_path() / 'models' / 'YOLO' / 'best.pt'
model = YOLO(model_path)
    
# Export the model to ONNX format
model.export(format='onnx')
