"""
This script reads images and their corresponding labels, and classes.txt from workspace directory
Creates a zip file, importable to cvat in new directory "export_to_cvat" inside workspace.
"""

import shutil
import yaml
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.utils.helpers import get_base_path, copy_and_merge

logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

base_path = get_base_path()
workspace = base_path / 'workspace'

if not workspace.exists():
    logger.error('Workspace directory not found')
    sys.exit(1)
#Paths
class_names = workspace / 'classes.txt'

if not class_names.exists():
    logger.error(f"'classes.txt' file not found in {workspace}")
    sys.exit(1)


save_dir = workspace / 'export_to_cvat'

Path.mkdir(save_dir, exist_ok=True)

label_path = save_dir / 'labels/train'
image_path = save_dir / 'images/train'

Path.mkdir(label_path, exist_ok=True, parents=True)
Path.mkdir(image_path, exist_ok=True, parents=True)

images_dir = workspace / 'images'

is_images = images_dir.is_dir() and any(
    path.suffix.lower() in ['.png', '.jpg', '.jpeg'] for path in images_dir.iterdir()
)

if not is_images:
    logger.error(f'No images found in {images_dir}')
    sys.exit(1)

images = [p for p in images_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
labels = [p.name for p in Path.iterdir(workspace / 'labels') if p.suffix.lower() == '.txt']

logger.info(f'Found {len(images)} images and {len(labels)} labels...')

for image_file in [p for p in images_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]:
    label_name = image_file.stem + '.txt'
    if label_name not in labels:
        continue
    copy_and_merge(Path(workspace / 'images' / image_file.name), Path(image_path / image_file.name))
    copy_and_merge(Path(workspace / 'labels' / label_name), Path(label_path / label_name))
    # path = Path('data/images/train/images', image_file.name)
    # with open(Path(save_dir,'train.txt'), 'a') as train:
    #     train.write(f'{path.as_posix()}\n')

#Extract class names to create yaml config
with open(class_names, 'r') as class_names_file:
    classes = []
    for line in class_names_file.readlines():
      if len(line.strip()) == 0: continue
      classes.append(line.strip())

data = {
    'names': {key:value for key, value in enumerate(classes)},
    'path': '.',
    'train': 'images'
}

# Write data to YAML file
with open(Path(save_dir, 'data.yaml'), 'w') as f:
    yaml.dump(data, f, sort_keys=False)

logger.info('Compressing data...')
archive = save_dir / 'cvat_export.zip'
archive_temp = save_dir.parent / 'cvat_export.zip'

if archive.exists():
    archive.unlink()

shutil.make_archive(str(archive_temp.with_suffix('').resolve()), 'zip', save_dir)

for item in save_dir.iterdir():
    if item.is_dir():
        shutil.rmtree(item)
    else:
        item.unlink()

archive_temp.replace(archive)
logger.info(f'Exported successfully to: {save_dir}')