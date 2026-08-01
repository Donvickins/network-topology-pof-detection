"""
This merges the converted bbox to segmentation
into other existing segmentation labels.
"""

import os
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.utils.helpers import get_base_path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')

base_path = get_base_path()
workspace = base_path / 'workspace' / 'merge'
detect_label_dir = workspace / 'bbox_2_segment'
segment_label_dir = workspace / 'segment_label'
merged_label_dir = workspace / 'all_labels_in_seg_form'

if not os.path.exists(workspace):
    logger.warning(f'Ensure workspace directory exists in {workspace}')
    sys.exit(1)

if not os.path.exists(detect_label_dir):
    logger.warning('Ensure you have exported bounding box to segmentation')
    logger.info('Run script bbox_2_seg.py, before this script')
    sys.exit(1)
    
if not os.path.exists(segment_label_dir):
    logger.warning('No segmentation directory found, using only bbox_2_segment directory')

os.makedirs(merged_label_dir, exist_ok=True)

for filename in os.listdir(detect_label_dir):
    if filename.endswith('.txt'):
        detect_file_path = os.path.join(detect_label_dir, filename)
        save_path = os.path.join(merged_label_dir, filename)

        with open(detect_file_path, 'r') as detect_label_file:
            detect_content = detect_label_file.read()

        segment_content = ""
        # If the segmentation directory exists, try to find and read the corresponding label file
        if os.path.exists(segment_label_dir):
            segment_file_path = os.path.join(segment_label_dir, filename)
            if os.path.exists(segment_file_path):
                with open(segment_file_path, 'r') as segment_label_file:
                    segment_content = segment_label_file.read()

        with open(save_path, 'w') as merged_file:
            merged_file.write(detect_content + segment_content)

logger.info("All files merged successfully")