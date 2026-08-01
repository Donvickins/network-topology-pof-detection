import sys
from ultralytics.data.converter import yolo_bbox2segment
import os
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.utils.helpers import get_base_path

BASE_DIR = get_base_path()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')

workspace = BASE_DIR / 'workspace' / 'merge'
bbox_dir = workspace / 'bbox_2_segment'
images_dir = workspace / 'images'

sam_b_model_path = BASE_DIR / 'models' / 'sam_b.pt'

if not os.path.exists(workspace):
    logger.warning(f'Ensure workspace directory exists in {workspace}')
    sys.exit(1)

if not os.path.exists(images_dir):
    logger.warning('Ensure images directory exists in workspace directory')
    sys.exit(1)
else:
    has_images = any(
        path.suffix.lower() in ['.png', '.jpg'] for path in Path(images_dir).rglob('*')
    )
    if not has_images:
        logger.warning('No images files found in the directory...')
        sys.exit(1)

if not os.path.exists(bbox_dir):
    os.makedirs(bbox_dir)

yolo_bbox2segment(
    im_dir= str(workspace),
    save_dir= str(os.path.join(workspace, 'bbox_2_segment')),
    sam_model=str(sam_b_model_path)
)