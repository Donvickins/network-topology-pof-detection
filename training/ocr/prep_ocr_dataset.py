"""
This script automate OCR dataset prep process.
This script extracts site id from network topology images and creates txt files for them
in preparation for training tesseract OCR engine

Find the prepared data in workspace/OCR/data/pof_ocr-ground-truth
"""

import logging
import sys
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.utils.helpers import get_base_path

PROJECT_BASE = get_base_path()

from core.utils.helpers import get_class_name, get_site_id_from_image, site_id_2_binary, extract_text
from core.utils.node_type_config import NODE_TYPE
from core.utils.helpers import move_and_merge

logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

base_path = PROJECT_BASE
workspace = base_path / 'workspace'
image_dir = workspace / 'images'
yolo_model_path = base_path / 'models' / 'YOLO' / 'best.pt'
save_img_dir = workspace / 'OCR' / 'images'

if not workspace.exists():
    logger.error('Workspace directory not found')
    sys.exit(0)

if not yolo_model_path.exists():
    logger.error(f'YOLO model not found in {yolo_model_path.parent}')
    sys.exit(0)

if not image_dir.is_dir() or not any(img.suffix.lower() in ['.png', '.jpg', '.jpeg'] for img in image_dir.iterdir()):
    logger.error(f'No images found in {image_dir}')
    sys.exit(0)

Path.mkdir(save_img_dir, exist_ok=True)
yolo_model = YOLO(yolo_model_path)

results = yolo_model.predict(
    source=image_dir,
    save=False,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

for idx, result in enumerate(results):
    for key, item in enumerate(result.boxes.data):
        class_id = item[-1]
        class_name = get_class_name(result=result, c_id=class_id)
        bbox = result.boxes.data[key][:4]
        image_name = f'image_{idx}{key}.png'
        txt_file = f'image_{idx}{key}.gt.txt'

        if any(class_name.startswith(p) for p in NODE_TYPE):
            site_id_image =  get_site_id_from_image(image_path=result.path, node_bbox=bbox)
            if isinstance(site_id_image, str) or site_id_image is None:
                continue

            site_id_binary = site_id_2_binary(site_id_image)
            cv2.imwrite(f'{str(save_img_dir)}/{image_name}', site_id_binary)
            label = Path(save_img_dir/txt_file)
            label.touch()
            with open(label, 'w') as f:
                f.write(extract_text(site_id_binary))


logger.info(f'Images saved to {save_img_dir}')

save_img_dir = workspace / 'OCR' / 'images'
data_dir = save_img_dir.parent / 'data' / 'pof_ocr-ground-truth'

if not save_img_dir.exists() or not any(img.suffix.lower() in ['.png', '.jpg', '.jpeg'] for img in save_img_dir.iterdir()):
    logger.error(f'No images found in {save_img_dir}')
    sys.exit(0)

Path.mkdir(data_dir, exist_ok=True, parents=True)

images = [file for file in save_img_dir.iterdir() if file.suffix.lower() in ['.png', '.jpg', '.jpeg']]
labels = [file.name for file in save_img_dir.iterdir() if file.suffix.lower() == '.txt']

logger.info(f"Found {len(images)} images and {len(labels)} labels")
logger.info('Sorting Images and label...')

for image_file in images:
    label_file = image_file.stem + '.gt.txt'
    if label_file in labels:
        # Define paths for the source image and label
        src_img_path = save_img_dir / image_file.name
        src_label_path = save_img_dir / label_file

        # Define destination paths
        dst_img_path = data_dir / image_file.name
        dst_label_path = data_dir / label_file

        # Move the image and its corresponding label file
        move_and_merge(src_img_path, dst_img_path)
        move_and_merge(src_label_path, dst_label_path)

logger.info('Sorting Completed Successfully...')
logger.info(f'Check sorted data in {data_dir}')

