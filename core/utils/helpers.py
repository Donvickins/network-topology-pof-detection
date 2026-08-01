import pytesseract
import logging
import cv2
import sys
import torch
import shutil
import numpy as np
from scipy.spatial.distance import cdist
from typing import Union
from pathlib import Path
from fuzzywuzzy import fuzz
from core.utils.exception_handler import InvalidImageException
from core.utils.node_type_config import NODE_TYPE, COLOR_MAP, FUZZY_PERCENTAGE

logger = logging.getLogger(__name__)
# --- Constants and Mappings ---
TYPE_MAP = {node_type: [1 if i == j else 0 for i in range(len(NODE_TYPE))] for j, node_type in enumerate(NODE_TYPE)}

MAX_DIST_THRESH = 50

def get_base_path() -> Path:
    """
    Determines the base path of the application, whether running as a script or a frozen executable.
    This is crucial for locating bundled resources like Tesseract.
    """
        
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # PyInstaller one-file mode
        return Path(sys._MEIPASS)
    elif Path(sys.argv[0]).resolve().suffix.lower() == '.exe':
        # Nuitka or other .exe: sys.argv[0] is the path to the executable.
        return Path(sys.argv[0]).resolve().parent
    else:
        # Running as a normal Python script
        return Path(__file__).resolve().parents[2]

# --- Tesseract Configuration ---
# Use a bundled Tesseract executable if available, otherwise fall back to system PATH.
# This ensures the custom model 'pof_ocr' is found in the bundled 'tessdata' directory.

_tesseract_configured = False

def _configure_tesseract() -> None:
    global _tesseract_configured
    if _tesseract_configured:
        return
    base_path = get_base_path()
    tesseract_path = base_path / "tesseract" / "tesseract.exe"
    if tesseract_path.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)
    else:
        logger.warning(f"Bundled Tesseract not found at '{tesseract_path}'. Falling back to system PATH.")
    _tesseract_configured = True

def extract_text(img) -> str | None:
    """
    Performs OCR using Tesseract

    We use a custom configuration to specify character set and OCR mode

    *custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'*
    Args:
        img (cv2.Mat): OpenCV Image

    Returns:
        str | None: A string containing the extracted data or None if OCR fails.
    """
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'

    _configure_tesseract()

    try:
        text = pytesseract.image_to_string(img, config=custom_config, lang='pof_ocr')
        # Clean up the output by stripping whitespace and newlines
        text_s = text.strip()
        idx = text_s.find('_')
        if idx == -1:
            idx = text_s.find('-')
            if idx == -1:
                id = text_s
            else:
                id = text_s[:idx]
        else:
            id = text_s[:idx]

        if id:
            return id
        else:
            return 'invalid'

    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract is not installed or not in your PATH. Please install it.")
        raise


def site_id_2_binary(img) -> Union[np.ndarray, None]:
    """
    This converts an image into black and white color, processes the image to be used for OCR

    :param
        img: This is an OpenCV image
    :return:
        np.ndarray | None: This is the binary form of the input image
    """
    if img is None:
        logger.error("Invalid Site Id Image")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 38, 120])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.resize(mask, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    return mask

def get_site_id_from_image(image_path: Union[str, Path, np.ndarray], node_bbox) -> Union[None, np.ndarray]:
    """
    Crops the site ID from a node image and performs OCR.
    Args:
        image_path (str): The path to the full topology image.
        node_bbox (list): Bounding box of the detected node in the format [x_min, y_min, x_max, y_max].

    Returns:
        np.ndarray | None : The Site ID from the image or String 'out of bounds' if its out of bounds of image, or None if image does not exist or is invalid.
    """
    # Load the image
    img = None

    if isinstance(image_path, (str, Path)):
        img = cv2.imread(str(image_path))
    elif isinstance(image_path, np.ndarray):
        img = image_path

    if img is None:
        logger.error(f"Could not load image")
        return None

    x_min, y_min, x_max, y_max = [int(coord) for coord in node_bbox]

    if y_min < y_max and x_min < x_max:
        detected_image = img[y_min:y_max, x_min:x_max]
    else:
        logger.warning("Detected object region is out of bounds or invalid.")
        return None

    row_start_raw = y_max + 1
    row_end_raw = row_start_raw + 22
    col_start_raw = x_min - 30
    col_end_raw = x_max - 1

    row_start = row_start_raw
    row_end = row_end_raw
    col_start = col_start_raw
    col_end = col_end_raw

    img_height, img_width = img.shape[:2]
    if (0 <= row_start < img_height and 0 <= row_end <= img_height and 0 <= col_start < img_width and 0 <= col_end <= img_width and
    row_start < row_end and col_start < col_end):
        site_id_image = img[row_start:row_end, col_start:col_end]
    else:
        logger.warning("Site ID region is out of bounds or invalid.")
        return None

    if site_id_image.size == 0:
        logger.warning("Cropped image is empty. Bounding box may be invalid.")
        return None

    return site_id_image

def get_site_id_from_node(image_path: Union[str, Path, np.ndarray], node_bbox) -> str | None:
    """
    Crops the site ID from a node image and performs OCR.
    Args:
        image_path (str): The path to the full topology image.
        node_bbox (list): Bounding box of the detected node in the format [x_min, y_min, x_max, y_max].

    Returns:
        str | None: The extracted site ID, or None if OCR fails.
    """
    site_id_image = get_site_id_from_image(image_path, node_bbox)

    if site_id_image is None or site_id_image.size == 0:
        return 'invalid'

    site_id_binary_image = site_id_2_binary(site_id_image)
    txt = extract_text(site_id_binary_image)
    return txt

def get_class_name(result, c_id) -> str | None:
    """
    Returns class name for any given valid class id

    Args:
        result (List): YOLO model result
        c_id (int): class id

    Returns:
        str | None: class name if it exists or None if it doesn't exist
    """
    for i, c_name in result.names.items():
        if int(c_id) == i:
            return c_name
    return None


def create_node_tensor(nodes: list, down_id: str = None) -> dict:
    """
    Create a node features, node centers and node ids from YOLO Detections

    Args:
        nodes (List): List of nodes detected by YOLO from the topology image
        down_id (str): Site Id of the faulty site
    Returns:
        dict: Dictionary containing node features, node centers, and node ids
    """
    if not nodes:
        logger.info('No nodes found, cannot create a graph.')
        raise InvalidImageException('No nodes found in image')

    node_features = []
    node_centers = []
    node_ids = []
    for node in nodes:
        feature = TYPE_MAP.get(node['type']) + COLOR_MAP.get(node['color'])
        is_down = 1 if down_id and fuzz.ratio(down_id, node['id']) >= FUZZY_PERCENTAGE else 0
        feature.append(is_down)
        node_centers.append(node['center'])
        node_ids.append(node['id'])
        node_features.append(feature)

    x = torch.tensor(node_features, dtype=torch.float)
    node_centers = np.array(node_centers)

    return {
        'x': x,
        'node_centers': node_centers,
        'node_ids': node_ids
    }


def create_edges_tensor(edges: list, node_centers: list) -> dict:
    """
    Creates edges tensors (connective lines in the topology images)

    Args:
        edges (List): List of connective lines or edges found in the topology image
        node_centers (List): List of node centers found in the topology image
    Returns:
        dict: Dictionary containing edge index (connective lines in topology image) and edge attributes
    """
    if len(node_centers) == 0:
        logger.warning("No node centers provided, cannot create edges.")
        return {
            'edge_index': torch.empty((2, 0), dtype=torch.long).contiguous(),
            'edge_attr': torch.empty((0, 6), dtype=torch.float)
        }

    if len(edges) == 0:
        logger.warning("No edges provided, returning empty edge tensors.")
        return {
            'edge_index': torch.empty((2, 0), dtype=torch.long).contiguous(),
            'edge_attr': torch.empty((0, 6), dtype=torch.float)
        }

    edge_list = []
    edge_attributes = []
    discarded_edges = 0

    for edge in edges:
        ep1, ep2 = edge['endpoints']
        dist1 = cdist([tuple(ep1)], node_centers, 'euclidean')
        dist2 = cdist([tuple(ep2)], node_centers, 'euclidean')
        src_idx = np.argmin(dist1)
        tgt_idx = np.argmin(dist2)

        if dist1[0, src_idx] <= MAX_DIST_THRESH and dist2[0, tgt_idx] <= MAX_DIST_THRESH and src_idx != tgt_idx:
            edge_list.extend([[src_idx, tgt_idx], [tgt_idx, src_idx]])
            attr = COLOR_MAP.get(edge['color'])
            edge_attributes.extend([attr, attr])
        else:
            discarded_edges += 1
            reason = (
                f"Distance to nearest nodes ({dist1[0, src_idx]:.2f}, {dist2[0, tgt_idx]:.2f}) exceeds threshold "
                f"{MAX_DIST_THRESH} or connects same node (src_idx={src_idx}, tgt_idx={tgt_idx})"
            )
            logger.debug(f"Edge discarded: {reason}")

    if discarded_edges > 0:
        logger.warning(f"Discarded {discarded_edges} links due to invalid connections.")

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long).contiguous()
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float) if edge_attributes else torch.empty((0, 6), dtype=torch.float)

    return {
        'edge_index': edge_index,
        'edge_attr': edge_attr
    }

def extract_data_from_YOLO(result: list, img: Union[str, Path, np.ndarray]) -> list:
    """
    Extracts detection data from YOLO model

    Args:
        result (list): Detections result from YOLO model
        img (str | Path | np.ndarray): Path to or binary image

    Returns:
         list: List of nodes and edges extracted from the image
    """

    if len(result) == 0 and not img:
        logger.info('Result and image cannot be empty')
        raise InvalidImageException('Image does not contain any valid nodes')

    boxes = result[0].boxes
    nodes = []
    edges = []

    for key, item in enumerate(boxes.data):
        class_id = item[-1]
        class_name = get_class_name(result=result[0], c_id=class_id)
        bbox = boxes.data[key][:4]
        center = boxes.xywh[key][0:2].cpu().detach().numpy().tolist()
        color = class_name.split('_')[-1]

        if class_name.startswith('Link'):
            x_min, y_min, x_max, y_max = boxes.xyxy[key].cpu().detach().numpy().tolist()
            edge = {'color': color, 'endpoints': [(x_min, y_min), (x_max, y_max)]}
            edges.append(edge)
        elif any(class_name.startswith(p) for p in NODE_TYPE):
            node_type = class_name.split('_')[0]
            site_id = get_site_id_from_node(image_path=img, node_bbox=bbox)
            if site_id == 'invalid' or not site_id:
                continue
            node = {'id': site_id, 'type': node_type, 'color': color, 'center': center}
            nodes.append(node)

    return [nodes, edges]

def copy_and_merge(src, dst):
    if not Path.exists(dst):
        shutil.copy2(src, dst)
    else:
        for item in Path.iterdir(src):
            src_path = Path(src) / item.name
            dst_path = Path(dst) / item.name
            if Path.is_dir(src_path):
                copy_and_merge(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

def move_and_merge(src: Path, dst: Path):
    """
    Recursively moves files and directories from src to dst.
    If dst does not exist, src is moved to dst.
    If dst exists, the contents of src are moved into dst, merging directories.
    """
    if not dst.exists():
        try:
            shutil.move(str(src), str(dst))
        except PermissionError as e:
            logger.error(f"Permission denied to move {src} to {dst}. The file might be in use. Error: {e}")
        return

    if not src.is_dir():
        try:
            shutil.move(str(src), str(dst))
        except PermissionError as e:
            logger.error(f"Permission denied to move file {src} into {dst}. The file might be in use. Error: {e}")
    else:
        for src_path in src.iterdir():
            dst_path = dst / src_path.name
            if src_path.is_dir():
                move_and_merge(src_path, dst_path)
            else:
                try:
                    shutil.move(str(src_path), str(dst_path))
                except PermissionError as e:
                    logger.error(f"Permission denied to move file {src_path} to {dst_path}. The file might be in use. Error: {e}")

if __name__ == '__main__':
    sys.exit(0)