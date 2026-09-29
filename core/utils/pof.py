"""
This contains the business logic of the application. This is where the 3 models work together to achieve POF prediction
"""

import logging
import sys
import io
import cv2
import torch
import numpy as np
from typing import Tuple, Union
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from core.utils import helpers as utils
from core.gnn.model import GNN
from core.utils.exception_handler import InvalidImageException, SiteIdNotFoundInImage, NoSiteId
from core.utils.constants import DEVICE, NUM_NODE_FEATURES, CONF_LEVEL, IMGSZ, MAX_IMAGE_PX

logger = logging.getLogger(__name__)


def prep_models(yolo_model_path: str, gnn_model_path: str) -> Union[Tuple[YOLO, GNN], None]:
    """
    Prepares the YOLO and GNN models and all its necessary configurations

    Args:
        yolo_model_path (Path): Path to the YOLO model.
        gnn_model_path (Path): Path to the GNN model.

    Returns:
        Union[Tuple[YOLO, GNN], None]: A Union containing a tuple of the YOLO and GNN models else None if loading fails.
    """
    base_path = utils.get_base_path()

    if not (base_path/ yolo_model_path).exists() or not (base_path/ gnn_model_path).exists():
        logger.warning(f'Either YOLO or GNN model does not exist in: {yolo_model_path} and {gnn_model_path}')
        return None

    try:
        yolo_model = YOLO(base_path / yolo_model_path)
        yolo_model.to(DEVICE)
        yolo_model.eval()

        logger.info(f'YOLO model loaded from: {yolo_model_path}')

        gnn_model = load_gnn_model( # from .pof.GNN.GModel import GNN
            model_path= base_path/ gnn_model_path,
            in_channels=NUM_NODE_FEATURES,
            hidden_channels=128,
            num_edge_features=6  # Edge color features
        )

        gnn_model.to(DEVICE)
        gnn_model.eval()
        logger.info(f'GNN model loaded from: {gnn_model_path}')

    except Exception as err:
        logger.error(f'Failed to load models: {err}')
        return None

    return yolo_model, gnn_model


def load_gnn_model(model_path, in_channels, hidden_channels, num_edge_features) -> GNN:
    """
    Loads the trained GNN model and sets it to evaluation mode.

    Args:
        model_path (Path): Path to the trained GNN model.
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.

    Returns:
        GNN: The loaded GNN model.
    """
    if not model_path.exists():
        logger.error('GNN model path does not exist')
        raise FileNotFoundError(f'GNN model path does not exist in: {model_path}')

    model = GNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_edge_features=num_edge_features
    )

    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=DEVICE))
    model.eval()
    return model


def interpret_pof_predictions(pof_logits, has_pof_logit, node_ids) -> Tuple[str, float]:
    """
    Interprets the POF prediction output, considering indeterminate cases.

    Args:
        pof_logits (torch.Tensor): Predicted POF logits.
        has_pof_logit (torch.Tensor): Predicted has_pof logits.
        node_ids (list): List of node IDs.

    Returns:
        Tuple[str, float]: A tuple containing the predicted POF site ID and its probability.
    """
    has_pof_prob = torch.sigmoid(has_pof_logit).item()
    if has_pof_prob < 0.5:
        return 'indeterminate', has_pof_prob
    probs = torch.sigmoid(pof_logits)
    pred_idx = torch.argmax(probs).item()
    pred_site_id = node_ids[pred_idx]
    pred_prob = probs[pred_idx].item()
    return pred_site_id, pred_prob

def pof(image, down_id: str, yolo_model, gnn_model) -> Tuple[str, float]:
    """
    Main function to process a network topology image and predict the point of failure.

    Args:
        image (bytes): Image to be processed
        down_id (str): Site Id down to be processed
        yolo_model (YOLO): YOLO model
        gnn_model (GNN): GNN model

    Returns:
        Tuple[str, float]: A tuple containing the predicted POF site ID and its probability.
    """
    # Check if image exists
    if image is None:
        logger.error('Invalid image')
        raise InvalidImageException('Image invalid')

    if not down_id:
        logger.error('Empty or invalid site ID')
        raise NoSiteId('Empty or invalid site ID')

    try:
        with Image.open(io.BytesIO(image)) as pil:
            if max(pil.size) > MAX_IMAGE_PX:
                logger.error(f'Image dimensions {pil.size} exceed {MAX_IMAGE_PX}px limit')
                raise InvalidImageException(f'Image dimensions exceed {MAX_IMAGE_PX}px limit')
            image = cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2BGR)
    except InvalidImageException:
        raise
    except Exception:
        logger.error('Invalid image')
        raise InvalidImageException('Image is not valid')

    if image.size == 0:
        logger.error('Invalid image')
        raise InvalidImageException('Image is not valid')

    # Run YOLO model on the image
    yolo_data = yolo_model.predict(source=image, save=False, conf=CONF_LEVEL, verbose=False, imgsz=IMGSZ, device=DEVICE)
    result = yolo_data[0]

    # class-wise counts
    cls_names = result.names
    cls_ids   = result.boxes.cls.int().cpu().tolist()
    counts = {}
    for cid in cls_ids:
        name = cls_names[cid]
        counts[name] = counts.get(name, 0) + 1

    count_str = ", ".join(f"{v} {k}" for k, v in counts.items())
    log_line = (
        f"Detections: {count_str}"
    )
    logger.info(log_line)
    # Process results to extract nodes and edges
    nodes, edges = utils.extract_data_from_YOLO(yolo_data, image)
    if len(nodes) == 0:
        logger.error('No nodes found in image or YOLO failed to process image')
        raise InvalidImageException('No nodes found in image')

    # Create graph tensors
    node_data = utils.create_node_tensor(nodes, down_id)
    edge_data = utils.create_edges_tensor(edges, node_data['node_centers'])

    # Validate graph
    if len(node_data['node_ids']) == 0:
        logger.error('No valid nodes extracted from image')
        raise InvalidImageException('No valid nodes extracted from image')

    if not isinstance(edge_data['edge_index'], torch.Tensor) or edge_data['edge_index'].shape[0] != 2:
        logger.error(f'Invalid edge_index: {edge_data["edge_index"]}')
        raise InvalidImageException('Invalid edge index')

    if edge_data['edge_index'].size(1) == 0:
        logger.warning('No valid edges detected in the graph, proceeding with isolated nodes.')

    # Fuzzy matching for down_id (single source: utils.match_down_id)
    matches = utils.match_down_id(down_id, node_data['node_ids'])

    if len(matches) == 0:
        logger.error(f'Site down with ID: "{down_id}" not found in image')
        raise SiteIdNotFoundInImage(f'Site down with ID: "{down_id}" not found in image')

    runner_ups = ", ".join(f"{m['id']} ({m['score']}%)" for m in matches[:3])
    logger.info(f'down_id "{down_id}" top matches: {runner_ups}')

    # Feed extracted nodes and edges to the GNN
    with torch.no_grad():
        pof_logits, has_pof_logit = gnn_model(node_data['x'], edge_data['edge_index'], edge_data['edge_attr'])

    # Interpret predictions
    return interpret_pof_predictions(pof_logits, has_pof_logit, node_data['node_ids'])

if __name__ == "__main__":
    sys.exit(0)