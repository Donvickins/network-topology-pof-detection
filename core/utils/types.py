from enum import Enum
from typing import NamedTuple
from torch import Tensor
import numpy as np

class OCR(Enum):
    OCR_EMPTY = 'No text was extracted'
    TESSERACT_NOT_INSTALLED = 'Tesseract is not installed or not in your PATH'
    TESSERACT_ERROR = 'Tesseract failed to process image'

class IMAGE(Enum):
    OUT_OF_BOUNDS = 'Site id is out of bounds or invalid'
    EMPTY_INPUT_IMAGE = 'Image is empty or invalid'
    EMPTY_CROPED_IMAGE = 'Cropped site id image is empty'
    INVALID_FILE_PATH = 'File path is invalid or does not exist'

class VALIDATION(Enum):
    EMPTY_FIELD = 'empty field'
    INVALID_IMAGE = 'invalid base64 image'
    IMAGE_TOO_LARGE = 'image dimensions exceed allowed limit'
    INVALID_ID = 'invalid ID format'

class TextResult(NamedTuple):
    text: str | None
    code: OCR | None
    error: str | None


class CropSiteIdResult(NamedTuple):
    site_id_image: np.ndarray | None
    code: IMAGE | None
    error: str | None


class SiteIdResult(NamedTuple):
    site_id: str | None
    code: OCR | IMAGE | None
    error: str | None


class DownIdMatchResult(NamedTuple):
    score: float
    id: str | None


class NodeData(NamedTuple):
    x: Tensor
    node_centers: np.ndarray
    node_ids: list[str]


class EdgeData(NamedTuple):
    edge_index: Tensor
    edge_attr: Tensor
