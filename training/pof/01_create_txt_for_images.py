"""
This utility script prepares txt files containing site_id down with corresponding pof predictions.
This is done in preparation for a personnel to fill in the pof per image.
This will be used as ground truth for training of the GNN model

This script assumes that workspace directory is on the project root
If you wish to run this script outside current directory. You need to modify workspace directory path to match
you desired directory.

Note: This script assumes the site_down id is in the file name bearing the structure: topology_AK0031_20250704_191908.png

The output of the operation will be saved workspace/pof/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.utils.helpers import get_base_path

PROJECT_ROOT = get_base_path()

workspace = PROJECT_ROOT / 'workspace'
image_path = workspace / 'images'

if not image_path.is_dir() or not any(file.is_file() for file in image_path.glob('*.png')):
    print(f'Images folder does not exist or no PNG files in {image_path}')
    sys.exit(1)

pof_path = workspace / 'pof'
pof_path.mkdir(parents=True, exist_ok=True)

for image in image_path.glob('*.png'):
    site_id = image.stem.split('_')[1]
    text = f'down: {site_id}\npof: '
    file_name = f'{image.stem}.txt'
    file_path = pof_path / file_name
    file_path.touch()
    with open(file_path, 'w') as f:
        f.write(text)

print('Task completed successfully.')