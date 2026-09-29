from enum import Enum

class OCR(Enum):
    OCR_EMPTY = 'No text was extracted'
    TESSERACT_NOT_INSTALLED = 'Tesseract is not installed or not in your PATH'
    TESSERACT_ERROR = 'Tesseract failed to process image'

class IMAGE(Enum):
    OUT_OF_BOUNDS = 'Site id is out of bounds or invalid'
    EMPTY_INPUT_IMAGE = 'Image is empty or invalid'
    EMPTY_CROPED_IMAGE = 'Cropped site id image is empty'

class VALIDATION(Enum):
    EMPTY_FIELD = 'empty field'
    INVALID_IMAGE = 'invalid base64 image'
    IMAGE_TOO_LARGE = 'image dimensions exceed allowed limit'
    INVALID_ID = 'invalid ID format'