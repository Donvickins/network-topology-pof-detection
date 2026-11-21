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
from core.utils.types import OCR, IMAGE, TextResult, CropSiteIdResult, SiteIdResult, NodeData, EdgeData, DownIdMatchResult
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
        logger.warning(f"Bundled Tesseract not found. Falling back to system. Please ensure Tesseract is installed and in your PATH.")
    _tesseract_configured = True

def extract_text(img) -> TextResult:
    """
    Performs OCR using Tesseract

    We use a custom configuration to specify character set and OCR mode
    Which can be overitten in config.yaml

    *Default: custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'*
    Args:
        img (cv2.Mat): OpenCV Image

    Returns:
        TextResult: A named tuple containing the extracted text and error information.
    """
    custom_config = f'--oem {TESS_OEM} --psm {TESS_PSM} -c tessedit_char_whitelist={TESS_CHARSET}'
    langs = list(TESS_LANGS) if TESS_LANGS else ['eng']
    if 'eng' not in langs:
        langs.append('eng')

    _configure_tesseract()

    saw_tesseract_error = False
    for lang in langs:
        try:
            text = pytesseract.image_to_string(img, config=custom_config, lang=lang)
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract is not installed or not in your PATH. Please set it up.")
            return TextResult(text=None, code=OCR.TESSERACT_NOT_INSTALLED, error=OCR.TESSERACT_NOT_INSTALLED.value)
        except pytesseract.TesseractError as e:
            logger.warning(f"OCR with lang '{lang}' failed: {e}. Trying fallback.")
            saw_tesseract_error = True
            continue

        if text:
            text_s: str = text.strip()
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
                return TextResult(text=id, code=None, error=None)
    
    code = OCR.TESSERACT_ERROR if saw_tesseract_error else OCR.OCR_EMPTY
    return TextResult(text=None, code=code, error=code.value)


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

def crop_site_id_from_image(image_path: Union[str, Path, np.ndarray], node_bbox) -> CropSiteIdResult:
    """
    Crops a nodes' site ID from the topology image
    Args:
        image_path (Union[str, Path, np.ndarray]): The path to the full topology image.
        node_bbox (list): Bounding box of the detected node in the format [x_min, y_min, x_max, y_max].

    Returns:
        CropSiteIdResult: The result containing the cropped site_id image and error information.
    """
    # Load the image
    img = None
    result = {}

    if isinstance(image_path, (str, Path)):
        if not Path(image_path).resolve().is_file():
            return CropSiteIdResult(site_id_image=None, code=IMAGE.INVALID_FILE_PATH, error=IMAGE.INVALID_FILE_PATH.value)
        img = cv2.imread(str(image_path))
    elif isinstance(image_path, np.ndarray):
        img = image_path

    if img is None:
        return CropSiteIdResult(site_id_image=None, code=IMAGE.EMPTY_INPUT_IMAGE, error=IMAGE.EMPTY_INPUT_IMAGE.value)

    x_min, y_min, x_max, y_max = [int(coord) for coord in node_bbox]

    if y_min < y_max and x_min < x_max:
        pass
    else:
        return CropSiteIdResult(site_id_image=None, code=IMAGE.OUT_OF_BOUNDS, error=IMAGE.OUT_OF_BOUNDS.value)

    row_start = y_max + LABEL_DY_TOP
    row_end = row_start + LABEL_HEIGHT
    col_start = x_min - LABEL_DX_LEFT
    col_end = x_max - LABEL_DX_RIGHT

    img_height, img_width = img.shape[:2]
    if (0 <= row_start < img_height and 0 <= row_end <= img_height and 0 <= col_start < img_width and 0 <= col_end <= img_width and
    row_start < row_end and col_start < col_end):
        site_id_image = img[row_start:row_end, col_start:col_end]
    else:
        return CropSiteIdResult(site_id_image=None, code=IMAGE.OUT_OF_BOUNDS, error=IMAGE.OUT_OF_BOUNDS.value)

    if site_id_image.size == 0:
        return CropSiteIdResult(site_id_image=None, code=IMAGE.EMPTY_CROPED_IMAGE, error=IMAGE.EMPTY_CROPED_IMAGE.value)

    return CropSiteIdResult(site_id_image=site_id_image, code=None, error=None)

def get_site_id_from_node(image_path: Union[str, Path, np.ndarray], node_bbox) -> SiteIdResult:
    """
    Crops the site ID from a node image and performs OCR.
    Args:
        image_path (Union[str, Path, np.ndarray]): The path to the full topology image.
        node_bbox (list): Bounding box of the detected node in the format [x_min, y_min, x_max, y_max].

    Returns:
        SiteIdResult: A result object containing the extracted site id text, error information, or site id None and error message if OCR fails or errors occured
    """
    site_id_image, code, error = crop_site_id_from_image(image_path, node_bbox)
    if error is not None:
        return SiteIdResult(site_id=None, code=code, error=error)

    text, code, error = extract_text(site_id_img_2_binary(site_id_image))

    if error is None:
        return SiteIdResult(site_id=text, code=None, error=None)
    else:
        return SiteIdResult(site_id=None, code=code, error=error)

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


def match_down_id(down_id: str, node_ids: list) -> list[DownIdMatchResult]:
    """
    Scores every node id against down_id, returns matches >= FUZZY_PERCENTAGE best-first.
    Single source of fuzzy-match truth for is_down flags and down-site resolution.

    Returns:
        list: [DownIdMatchResult], sorted by score descending.
    """
    matches = [
        DownIdMatchResult(score=fuzz.ratio(down_id, node_id), id=node_id)
        for node_id in node_ids
    ]
    return sorted(
        (m for m in matches if m.score >= FUZZY_PERCENTAGE),
        key=lambda m: m.score, reverse=True,
    )


def create_node_tensor(nodes: list, down_id: str = None) -> NodeData:
    """
    Create a node features, node centers and node ids from YOLO Detections

    Args:
        nodes (List): List of nodes detected by YOLO from the topology image
        down_id (str): Site Id of the faulty site
    Returns:
        NodeData: Named tuple containing node features, node centers, and node ids
    """
    if not nodes:
        logger.info('No nodes found, cannot create a graph.')
        raise InvalidImageException('No nodes found in image')

    node_features = []
    node_centers = []
    node_ids = []
    matched_ids = {m.id for m in match_down_id(down_id, [node['id'] for node in nodes])}
    for node in nodes:
        feature = TYPE_MAP.get(node['type']) + COLOR_MAP.get(node['color'])
        is_down = 1 if node['id'] in matched_ids else 0
        feature.append(is_down)
        node_centers.append(node['center'])
        node_ids.append(node['id'])
        node_features.append(feature)

    x = torch.tensor(node_features, dtype=torch.float).to(DEVICE)
    node_centers = np.array(node_centers)

    return NodeData(
        x=x,
        node_centers=node_centers,
        node_ids=node_ids
    )


def create_edges_tensor(edges: list, node_centers: list) -> EdgeData:
    """
    Creates edges tensors (connective lines in the topology images)

    Args:
        edges (List): List of connective lines or edges found in the topology image
        node_centers (List): List of node centers found in the topology image
    Returns:
        EdgeData: Named tuple containing edge index (connective lines in topology image) and edge attributes
    """
    if len(node_centers) == 0:
        logger.warning("No node centers provided, returning empty edge tensors")
        return EdgeData(
            edge_index=torch.empty((2, 0), dtype=torch.long).contiguous().to(DEVICE),
            edge_attr=torch.empty((0, 6), dtype=torch.float).to(DEVICE)
        )

    if len(edges) == 0:
        logger.warning("No edges provided, returning empty edge tensors.")
        return EdgeData(
            edge_index=torch.empty((2, 0), dtype=torch.long).contiguous().to(DEVICE),
            edge_attr=torch.empty((0, 6), dtype=torch.float).to(DEVICE)
        )

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

    return EdgeData(
        edge_index=edge_index.to(DEVICE),
        edge_attr=edge_attr.to(DEVICE)
    )

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
            site_id, code, error = get_site_id_from_node(image_path=img, node_bbox=bbox)
            if code is not None:
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