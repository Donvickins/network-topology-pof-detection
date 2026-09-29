import pytesseract
import logging
import cv2
import sys
import torch
import shutil
import os
import numpy as np
from scipy.spatial.distance import cdist
from typing import Union
from pathlib import Path
from rapidfuzz import fuzz
from core.utils.exception_handler import InvalidImageException
from core.utils.node_type_config import NODE_TYPE, COLOR_MAP
from core.utils.constants import DEVICE, TESS_OEM, TESS_PSM, TESS_CHARSET, MAX_DIST_THRESH, FUZZY_PERCENTAGE, HSV_LOWER, HSV_UPPER, OCR_UPSCALE, TESS_LANGS, LABEL_DY_TOP, LABEL_HEIGHT, LABEL_DX_LEFT, LABEL_DX_RIGHT
from core.utils.enums import OCR, IMAGE
from core.utils.paths import get_base_path

logger = logging.getLogger(__name__)
# --- Constants and Mappings ---
TYPE_MAP = {node_type: [1 if i == j else 0 for i in range(len(NODE_TYPE))] for j, node_type in enumerate(NODE_TYPE)}


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

def extract_text(img) -> dict:
    """
    Performs OCR using Tesseract

    We use a custom configuration to specify character set and OCR mode
    Which can be overitten in config.yaml

    *Default: custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'*
    Args:
        img (cv2.Mat): OpenCV Image

    Returns:
        dict: A dict containing the extracted text and error None or text None, with error if OCR fails.
    """
    custom_config = f'--oem {TESS_OEM} --psm {TESS_PSM} -c tessedit_char_whitelist={TESS_CHARSET}'
    langs = list(TESS_LANGS) if TESS_LANGS else ['eng']
    if 'eng' not in langs:
        langs = langs + ['eng']
    result = {}

    _configure_tesseract()

    saw_tesseract_error = False
    for lang in langs:
        try:
            text = pytesseract.image_to_string(img, config=custom_config, lang=lang)
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract is not installed or not in your PATH. Please install it.")
            result['text'] = None
            result['code'] = OCR.TESSERACT_NOT_INSTALLED
            result['error'] = OCR.TESSERACT_NOT_INSTALLED.value
            return result
        except pytesseract.TesseractError as e:
            logger.warning(f"OCR with lang '{lang}' failed: {e}. Trying fallback.")
            saw_tesseract_error = True
            continue
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
            result['text'] = id
            result['code'] = None
            result['error'] = None
            return result

    result['text'] = None
    result['code'] = OCR.TESSERACT_ERROR if saw_tesseract_error else OCR.OCR_EMPTY
    result['error'] = result['code'].value
    return result


def site_id_img_2_binary(img) -> np.ndarray:
    """
    This converts an image into black and white color, processes the image to be used for OCR

    :param
        img: This is an OpenCV image
    :return:
        np.ndarray | None: This is the binary form of the input image
    """
    if img is None:
        logger.error("invalid site id image")
        raise InvalidImageException('invalid site id image')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(HSV_LOWER)
    upper = np.array(HSV_UPPER)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.resize(mask, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE, interpolation=cv2.INTER_LINEAR)
    return mask

def crop_site_id_from_image(image_path: Union[str, Path, np.ndarray], node_bbox) -> dict:
    """
    Crops a nodes' site ID from the topology image
    Args:
        image_path (str): The path to the full topology image.
        node_bbox (list): Bounding box of the detected node in the format [x_min, y_min, x_max, y_max].

    Returns:
        dict: The dictionary containing the cropped site_id image and error None or site_id None and error description if image does not exist or is invalid or any other error.
    """
    # Load the image
    img = None
    result = {}

    if isinstance(image_path, (str, Path)):
        img = cv2.imread(str(image_path))
    elif isinstance(image_path, np.ndarray):
        img = image_path

    if img is None:
        result['site_id_image'] = None
        result['code'] = IMAGE.EMPTY_INPUT_IMAGE
        result['error'] = IMAGE.EMPTY_INPUT_IMAGE.value
        return result

    x_min, y_min, x_max, y_max = [int(coord) for coord in node_bbox]

    if y_min < y_max and x_min < x_max:
        pass
        #detected_image = img[y_min:y_max, x_min:x_max]
    else:
        result['site_id_image'] = None
        result['code'] = IMAGE.OUT_OF_BOUNDS
        result['error'] = IMAGE.OUT_OF_BOUNDS.value
        return result

    row_start = y_max + LABEL_DY_TOP
    row_end = row_start + LABEL_HEIGHT
    col_start = x_min - LABEL_DX_LEFT
    col_end = x_max - LABEL_DX_RIGHT

    img_height, img_width = img.shape[:2]
    if (0 <= row_start < img_height and 0 <= row_end <= img_height and 0 <= col_start < img_width and 0 <= col_end <= img_width and
    row_start < row_end and col_start < col_end):
        site_id_image = img[row_start:row_end, col_start:col_end]
    else:
        result['site_id_image'] = None
        result['code'] = IMAGE.OUT_OF_BOUNDS
        result['error'] = IMAGE.OUT_OF_BOUNDS.value
        return result

    if site_id_image.size == 0:
        result['site_id_image'] = None
        result['code'] = IMAGE.EMPTY_CROPED_IMAGE
        result['error'] = IMAGE.EMPTY_CROPED_IMAGE.value
        return result

    result['site_id_image'] = site_id_image
    result['code'] = None
    result['error'] = None
    return result

def get_site_id_from_node(image_path: Union[str, Path, np.ndarray], node_bbox) -> dict:
    """
    Crops the site ID from a node image and performs OCR.
    Args:
        image_path (str): The path to the full topology image.
        node_bbox (list): Bounding box of the detected node in the format [x_min, y_min, x_max, y_max].

    Returns:
        dict: A dictionary containing the extracted site id text, error None or site id None and error message if OCR fails or errors occured
    """
    result = {}
    data = crop_site_id_from_image(image_path, node_bbox)
    site_id_image = data['site_id_image']

    if site_id_image is None or site_id_image.size == 0:
        result['site_id'] = None
        result['code'] = data['code']
        result['error'] = data['error']
        return result

    site_id_binary_image = site_id_img_2_binary(site_id_image)
    text_result = extract_text(site_id_binary_image)

    if text_result['code'] is None:
        result['code'] = None
        result['error'] = None
        result['site_id'] = text_result['text']
        return result
    else:
        result['code'] = text_result['code']
        result['error'] = text_result['error']
        result['site_id'] = None
        return result

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


def match_down_id(down_id: str, node_ids: list) -> list:
    """
    Scores every node id against down_id, returns matches >= FUZZY_PERCENTAGE best-first.
    Single source of fuzzy-match truth for is_down flags and down-site resolution.

    Returns:
        list: [{'score': int, 'id': str}], sorted by score descending.
    """
    if not down_id:
        return []
    matches = [
        {'score': fuzz.ratio(down_id, node_id), 'id': node_id}
        for node_id in node_ids
    ]
    return sorted(
        (m for m in matches if m['score'] >= FUZZY_PERCENTAGE),
        key=lambda m: m['score'], reverse=True,
    )


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
    matched_ids = {m['id'] for m in match_down_id(down_id, [n['id'] for n in nodes])}
    for node in nodes:
        feature = TYPE_MAP.get(node['type']) + COLOR_MAP.get(node['color'])
        is_down = 1 if node['id'] in matched_ids else 0
        feature.append(is_down)
        node_centers.append(node['center'])
        node_ids.append(node['id'])
        node_features.append(feature)

    x = torch.tensor(node_features, dtype=torch.float).to(DEVICE)
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
        logger.warning("No node centers provided, returning empty edge tensors")
        return {
            'edge_index': torch.empty((2, 0), dtype=torch.long).contiguous().to(DEVICE),
            'edge_attr': torch.empty((0, 6), dtype=torch.float).to(DEVICE)
        }

    if len(edges) == 0:
        logger.warning("No edges provided, returning empty edge tensors.")
        return {
            'edge_index': torch.empty((2, 0), dtype=torch.long).contiguous().to(DEVICE),
            'edge_attr': torch.empty((0, 6), dtype=torch.float).to(DEVICE)
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
        'edge_index': edge_index.to(DEVICE),
        'edge_attr': edge_attr.to(DEVICE)
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
        elif any(class_name.startswith(node) for node in NODE_TYPE):
            node_type = class_name.split('_')[0]
            data = get_site_id_from_node(image_path=img, node_bbox=bbox)
            site_id = data['site_id']
            error = data['error']
            if data['code'] is not None:
                logger.warning(error)
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


def is_demo_mode():
    return True if os.getenv('POF_MODEL_SOURCE', 'live') == 'demo' else False

if __name__ == '__main__':
    sys.exit(0)