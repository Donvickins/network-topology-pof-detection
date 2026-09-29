"""
Demo models for running the API without the private weights.

Active only when POF_MODEL_SOURCE=demo. Nothing returned in demo mode is a
real prediction: the detector is hardcoded boxes and the GNN has random
weights. Check /health (environment=demo) to see which mode is running.
"""

import logging
import torch
import cv2
import numpy as np

from core.gnn.model import GNN
from core.utils.constants import DEVICE, NUM_NODE_FEATURES
from core.utils.paths import get_base_path

logger = logging.getLogger(__name__)

# Fixed seed so demo output is identical on every boot.
# Seed 8 chosen because random weights then yield a decisive (non-indeterminate)
# answer on the bundled sample image. Still fake — never a real prediction.
DEMO_SEED = 8

# Canned detections. Coordinates must match the drawing in ensure_demo_image
# or the label crops will miss the text.
# Format per box: [x_min, y_min, x_max, y_max, confidence, class_id].
DEMO_BOXES = [
    [100.0, 80.0, 160.0, 140.0, 0.99, 0],   # ATN_Red node, label DEMO1
    [300.0, 80.0, 360.0, 140.0, 0.99, 1],   # RTN_Green node, label DEMO2
    [150.0, 100.0, 310.0, 120.0, 0.95, 2],  # Link_Blue edge between them
]
DEMO_NAMES = {0: 'ATN_Red', 1: 'RTN_Green', 2: 'Link_Blue'}
DEMO_NODE_LABELS = {0: 'DEMO1', 1: 'DEMO2'}

SAMPLE_REL_PATH = 'workspace/samples/demo_topology.png'


class DemoBoxes:
    """Mimics the ultralytics Boxes fields used in core/utils/helpers.py."""

    def __init__(self, data: torch.Tensor):
        self.data = data

    @property
    def xywh(self) -> torch.Tensor:
        x1, y1, x2, y2 = (self.data[:, i] for i in range(4))
        return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=1)

    @property
    def xyxy(self) -> torch.Tensor:
        return self.data[:, :4]

    @property
    def cls(self) -> torch.Tensor:
        return self.data[:, 5]


class DemoResult:
    """Mimics the ultralytics result fields used in pof()/helpers.py."""

    def __init__(self):
        self.boxes = DemoBoxes(torch.tensor(DEMO_BOXES, dtype=torch.float32))
        self.names = DEMO_NAMES


class DemoYOLO:
    """Hardcoded detector: ignores the photo, returns the canned boxes."""

    def predict(self, source=None, **kwargs):
        _ = (source, kwargs)
        return [DemoResult()]

    def to(self, device):
        _ = device
        return self

    def eval(self):
        return self


def ensure_demo_image():
    """
    Draws the sample topology matching DEMO_BOXES and returns its path.

    Labels sit in the exact strip core/utils/helpers.py crops
    (below each node box), in saturated red on white so the HSV filter
    keeps them. Upload this file to /pof with site_id DEMO1.
    """
    from core.utils.constants import LABEL_DY_TOP, LABEL_HEIGHT, LABEL_DX_LEFT, LABEL_DX_RIGHT

    sample_path = get_base_path() / SAMPLE_REL_PATH
    if sample_path.is_file():
        return sample_path

    canvas = np.full((220, 480, 3), 255, dtype=np.uint8)

    for class_id, (x1, y1, x2, y2, _, _) in enumerate(DEMO_BOXES[:2]):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (90, 90, 90), 2)

        strip_x1 = x1 - LABEL_DX_LEFT
        strip_x2 = x2 - LABEL_DX_RIGHT
        strip_y1 = y2 + LABEL_DY_TOP
        strip_y2 = strip_y1 + LABEL_HEIGHT
        label = DEMO_NODE_LABELS[class_id]

        scale = 0.8
        while scale > 0.3:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            if tw <= (strip_x2 - strip_x1 - 4) and th <= (LABEL_HEIGHT - 4):
                break
            scale -= 0.1
        cv2.putText(canvas, label, (strip_x1 + 2, strip_y2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2, cv2.LINE_AA)

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sample_path), canvas)
    return sample_path


def load_demo_models():
    """
    Builds demo models: hardcoded DemoYOLO + random-weight GNN.

    Returns:
        tuple: (DemoYOLO, GNN) — same shape as prep_models().
    """
    logger.warning('DEMO MODE: predictions are fake (hardcoded boxes, random GNN weights).')
    torch.manual_seed(DEMO_SEED)
    gnn_model = GNN(
        in_channels=NUM_NODE_FEATURES,
        hidden_channels=128,
        num_edge_features=6,
    )
    gnn_model.to(DEVICE)
    gnn_model.eval()
    sample_path = ensure_demo_image()
    logger.warning(f'DEMO MODE: upload {sample_path} to /pof with site_id DEMO1.')
    return DemoYOLO(), gnn_model
