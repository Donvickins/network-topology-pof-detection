"""
This script prepares dataset for training by gnn_trainer.py
"""

import sys
import cv2
from ultralytics import YOLO
import logging
from pathlib import Path
import torch
from torch_geometric.data import Data
from fuzzywuzzy import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.utils import helpers as utils
from core.utils.pof import IMGSZ, FUZZY_PERCENTAGE, CONF_LEVEL, DEVICE
from core.utils.helpers import get_base_path

PROJECT_ROOT = get_base_path()

logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

def create_graph_from_image_data(nodes, edges, pof_id, down_id):
    """Converts lists of nodes and edges into a PyTorch Geometric Data object."""
    if not nodes:
        logger.warning("No nodes found, cannot create a graph.")
        return None

    node_data = utils.create_node_tensor(nodes, down_id=down_id)
    edge_data = utils.create_edges_tensor(edges, node_data['node_centers'])

    if len(edge_data['edge_index'].shape) < 2 or edge_data['edge_index'].size(1) == 0:
        logger.warning("No valid edges found for the graph. Proceeding with isolated nodes.")
    # POF label
    pof_y = torch.zeros(len(nodes), dtype=torch.float)
    has_pof = 0

    # Find indices
    node_ids = node_data['node_ids']
    down_idx = next((i for i, nid in enumerate(node_ids) if fuzz.ratio(down_id, nid) >= FUZZY_PERCENTAGE), None)

    if down_idx is None:
        logger.warning(f"Down ID '{down_id}' not found in extracted nodes for this image, skipping graph creation.")
        return None

    pof_idx = next((i for i, nid in enumerate(node_ids) if fuzz.ratio(nid, pof_id) >= FUZZY_PERCENTAGE), None)

    if pof_idx is None and pof_id is not None:
        logger.warning(f"POF ID '{pof_id}' not found in extracted nodes, assuming no POF.")
        has_pof = 0
    elif pof_idx is not None:
        pof_y[pof_idx] = 1.0
        has_pof = 1

    return Data(
        x=node_data['x'],
        edge_index=edge_data['edge_index'],
        edge_attr=edge_data['edge_attr'],
        y=pof_y,
        has_pof=torch.tensor([has_pof], dtype=torch.float),
        node_ids=node_data['node_ids']
    )

def main():
    workspace = PROJECT_ROOT / 'workspace'
    images_dir = workspace / 'images'
    labels_dir = workspace / 'pof'
    processed_dir = workspace / 'processed'
    processed_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    model_file = Path(PROJECT_ROOT, 'models', 'YOLO','best.pt')
    model = YOLO(model_file)

    image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
    logger.info(f"Found {len(image_files)} images to process.")

    for image_path in image_files:
        logger.info(f"Processing {image_path.name}...")
        # Load corresponding label
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            logger.warning(f"No label file found for {image_path.name}, skipping.")
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()
            down_line = lines[0].strip() if lines else ""
            pof_line = lines[1].strip() if len(lines) > 1 else ""
            down_id = down_line.split(': ')[1] if down_line and ': ' in down_line else None
            pof_id = pof_line.split(': ')[1] if pof_line and ': ' in pof_line else None

        if not down_id:
            logger.warning(f"Invalid label format (missing down_id) for {image_path.name}, skipping.")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            continue

        results = model.predict(source=image_path, save=False, show_labels=False, show_conf=False, imgsz=IMGSZ,device=DEVICE)
        nodes, edges = utils.extract_data_from_YOLO(results, image_path)
        graph_data = create_graph_from_image_data(nodes, edges, pof_id, down_id)

        if graph_data:
            output_path = processed_dir / f"{image_path.stem}.pt"
            torch.save(graph_data, output_path)
            logger.info(f"Saved processed graph to {output_path}")
        else:
            logger.warning(f"Could not create graph for {image_path.name}, skipping.")

    logger.info("Data preparation complete.")

if __name__ == '__main__':
    main()