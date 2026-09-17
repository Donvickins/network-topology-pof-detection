import torch

NUM_NODE_FEATURES = 12  # 5 (type) + 6 (color)  + 1 (down_id)
CONF_LEVEL = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMGSZ = 1280