import logging
from pathlib import Path

import torch
import yaml

from core.utils.paths import get_base_path

logger = logging.getLogger(__name__)

def safe_int(value, fallback: int) -> int:
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return fallback

def safe_float(value, fallback: float) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return fallback

def safe_hsv(value, fallback: list, msg: None | str = None) -> list:
    if not isinstance(value, (list, tuple)):
        if msg and msg.strip():
            logger.warning(msg)
        return fallback

    if len(value) != 3:
        if msg and msg.strip():
            logger.warning(msg)
        return fallback

    try:
        return [int(float(x)) for x in value]
    except (ValueError, TypeError):
        return fallback

def safe_langs(value, fallback, msg=None) -> list:
    if not isinstance(value, (tuple, list)) or len(value) == 0:
        if msg and msg.strip():
            logger.warning(msg)
        return fallback

    return value

def safe_path(path: str, fallback: str) -> Path:
    return Path(path) if Path(path).is_relative_to(get_base_path()) else fallback

def safe_bool(value, fallback: bool):
    if isinstance(value, bool):
        return value
    return fallback
    
config = {}
config_path = get_base_path() / 'config.yaml'
try:
    with open(config_path, 'r') as file:
        content = yaml.safe_load(file)
        if content:
            config = content
except Exception:  # noqa: BLE001
    logger.warning("Failed to load config.yaml, using defaults")


raw_nodes = config.get('model', {}).get('num_node_features', 12)  # 5 (type) + 6 (color)  + 1 (down_id)
NUM_NODE_FEATURES = safe_int(raw_nodes, 12)

IMGSZ = safe_int(config.get('model', {}).get('imgsz', 1280), 1280)
CONF_LEVEL = safe_float(config.get('thresholds', {}).get('conf_level', 0.1), 0.1)

device_options = ['auto', 'cpu', 'cuda']
raw_device = str(config.get('model', {}).get('device', 'cuda')).lower()
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu') if (raw_device not in device_options or raw_device == 'auto') else raw_device

MAX_DIST_THRESH = safe_int(config.get('thresholds', {}).get('max_dist_thresh', 50), 50)
FUZZY_PERCENTAGE = safe_int(config.get('thresholds', {}).get('fuzzy_percentage', 70), 70)
HSV_UPPER = safe_hsv(config.get('thresholds', {}).get('hsv_upper', [179, 255, 255]), [179, 255, 255], "invalid hsv input, using defaults")
HSV_LOWER = safe_hsv(config.get('thresholds', {}).get('hsv_lower', [0, 38, 120]), [0, 38, 120], "invalid hsv input, using defaults")

TESS_OEM = safe_int(config.get('tesseract', {}).get('oem', 3), 3)
TESS_PSM = safe_int(config.get('tesseract', {}).get('psm', 6), 6)

langs = config.get('tesseract', {}).get('langs', ['pof_ocr', 'eng'])
TESS_LANGS = safe_langs(langs, ['pof_ocr', 'eng'], 'Invalid language in config, using defaults')
TESS_CHARSET = config.get('tesseract', {}).get('charset', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_')
OCR_UPSCALE = safe_float(config.get('tesseract', {}).get('ocr_upscale', 3.0), 3.0)

SAVE_DIR = safe_path(config.get('inference', {}).get('save_dir', 'workspace/received_images'), 'workspace/received_images')
MAX_BODY_SIZE_MB = safe_int(config.get('server', {}).get('max_body_size_mb', 15), 15)
MAX_IMAGE_PX = safe_int(config.get('server', {}).get('max_image_px', 4500), 4500)
SAVE_RECEIVED_IMAGES = safe_bool(config.get('server', {}).get('save_received_images', True), True)

REQUEST_TIMEOUT_S = safe_int(config.get('server', {}).get('request_timeout_s', 25), 25)
HOST = config.get('server', {}).get('host', '127.0.0.1')
PORT = safe_int(config.get('server', {}).get('port', 5500), 5500)

LABEL_DY_TOP = 1
LABEL_HEIGHT = 22
LABEL_DX_LEFT = 30
LABEL_DX_RIGHT = 1