import logging
from uuid import uuid4
from pathlib import Path
from core.utils.constants import SAVE_DIR
from core.utils.paths import get_base_path

logger = logging.getLogger(__name__)

SAVE_DIR =  get_base_path() / SAVE_DIR
SAVE_DIR.mkdir(exist_ok=True, parents=True)


def save_received_image(data, file_name: str):
    try:
        image_path = SAVE_DIR / f'{uuid4()}-{file_name}.png'
        if image_path.is_symlink():
            logger.info(f'Symlink detected for id {file_name}')
            return
        
        image_path = image_path.resolve()
        if SAVE_DIR not in image_path.parents:
            logger.info(f'Attempt to write outside save directory prohibited')
            return
        
        with open(image_path, 'wb') as file:
            file.write(data)
    except Exception as e:
        logger.error(f'Failed to save image for order id: {file_name}. Reason: {e}')