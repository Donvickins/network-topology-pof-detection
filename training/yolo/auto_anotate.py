"""
This script auto annotates/label new images and prepares labels that may be imported to CVAT AI in preparation for supervised learning
and finetuning of YOLO model
"""

import sys
import numpy as np
import logging
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.utils.helpers import get_base_path
from ultralytics.data.annotator import auto_annotate

BASE_DIR = get_base_path()

logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

LINE_CLASS_ID = [6,7,8,9,20]

WORKSPACE_DIR = Path('workspace') if Path('workspace').exists() else BASE_DIR / 'workspace'
IMAGE_DIR = WORKSPACE_DIR / 'images'
OUTPUT_DIR = WORKSPACE_DIR / 'auto_labelled'
MODEL_PATH = BASE_DIR / 'models' / 'YOLO' / 'best.pt'

labelled_image_dir = OUTPUT_DIR / 'labelled_images'
auto_labelled_dir = OUTPUT_DIR / 'labels'
temp_annotated = OUTPUT_DIR / 'temp'

if not IMAGE_DIR.is_dir():
    logger.info('Images directory does not exist')
    sys.exit(0)

total_images = list(IMAGE_DIR.glob('*.jpg')) + list(IMAGE_DIR.glob('*.png'))
if not total_images:
    logger.info('Images directory does not contain any image files')
    sys.exit(0)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
labelled_image_dir.mkdir(parents=True, exist_ok=True)

auto_annotate(
    data=str(IMAGE_DIR),
    det_model=str(MODEL_PATH),
    sam_model='sam_b.pt',
    output_dir=str(auto_labelled_dir),
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

def simplify_polygon(coords, epsilon=0.002):
    """
        Clean Ramer-Douglas-Peucker
        This algorithm takes the multipoint polygon annotations of YOLO segmentation model and SAM
        and create a more simplified polygon to be imported to CVAT AI.
        This makes it easier for a human to correct errors in the models annotation
    """

    points = np.array(coords).reshape(-1, 2)
    if len(points) < 4:
        return points.flatten().tolist()

    def _rdp(pts):
        if len(pts) <= 2:
            return pts
        # Line from start to end
        start, end = pts[0], pts[-1]
        line_vec = end - start
        # Distances from line
        dists = np.abs(np.cross(line_vec, pts - start)) / np.linalg.norm(line_vec)
        max_idx = np.argmax(dists)
        if dists[max_idx] > epsilon:
            left = _rdp(pts[:max_idx+1])
            right = _rdp(pts[max_idx:])
            return np.vstack((left[:-1], right))
        return np.array([start, end])

    simplified = _rdp(points)
    # Close polygon
    if len(simplified) > 0 and not np.allclose(simplified[0], simplified[-1]):
        simplified = np.vstack((simplified, simplified[0]))
    return simplified.flatten().tolist()

logger.info("Simplifying SAM polygons for CVAT...")

simplified_dir = OUTPUT_DIR / "labels_simplified_for_cvat"
simplified_dir.mkdir(exist_ok=True)

for txt_file in auto_labelled_dir.glob("*.txt"):
    lines_out = []
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            coordinates = [float(x) for x in parts[1:]]

            # Use aggressive simplification for lines, gentle for nodes
            epsilon = 0.0003 if cls_id in LINE_CLASS_ID else 0.002
            clean_coords = simplify_polygon(coordinates, epsilon=epsilon)

            if len(clean_coords) >= 6 and len(clean_coords) % 2 == 0:
                lines_out.append(f"{cls_id} {' '.join(map(str, clean_coords))}")

    if lines_out:
        out_path = simplified_dir / txt_file.name
        with open(out_path, "w") as f:
            f.write("\n".join(lines_out) + "\n")

logger.info(f'Completed Successfully: Check: {OUTPUT_DIR}')