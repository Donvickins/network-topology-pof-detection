"""
This contains api routes for the application. New routes may be added if need be
"""
import logging
logger = logging.getLogger(__name__)
logger.info('Loading modules...')

import sys
import base64
import binascii
from fastapi import FastAPI, status, Request
from core.utils.schema import Request as pofRequest, Response as pofResponse
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from core.utils.pof import pof, prep_models
from pathlib import Path
from core.gnn.model import GNN
from core.utils.helpers import get_base_path
from ultralytics import YOLO
from core.utils.exception_handler import InvalidImageException, SiteIdNotFoundInImage, NoSiteId

logger.info('Modules loaded successfully')

app = FastAPI()

base_path = get_base_path()

yolo_model_path = base_path / 'models/YOLO/best.pt'
gnn_model_path = base_path / 'models/GNN/best.pt'

try:
    yolo_model, gnn_model = prep_models(yolo_model_path, gnn_model_path)
except Exception as e:
    sys.exit(1)

if not isinstance(yolo_model, YOLO) or not isinstance(gnn_model, GNN):
    logger.error('Failed to load models')
    sys.exit(1)

save_dir = base_path / 'workspace/received_images'
save_dir.mkdir(exist_ok=True, parents=True)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"status": "error", "message": "Validation error", "detail": exc.errors()}
    )

@app.exception_handler(NoSiteId)
async def no_site_id_exception_handler(request: Request, exc: InvalidImageException):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"status": "error", "message": "No site id was provided"}
    )

@app.exception_handler(InvalidImageException)
async def image_exception_handler(request: Request, exc: InvalidImageException):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"status": "error", "message": f"Invalid image provided: {exc.details}"}
    )

@app.exception_handler(SiteIdNotFoundInImage)
async def site_id_exception_handler(request: Request, exc: SiteIdNotFoundInImage):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"status": "error", "message": "Site ID not found in image"}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "message": "An unexpected error occurred on the server."},
    )

@app.post('/pof')
async def check_pof(request: pofRequest):
    try:
        image_bytes = base64.b64decode(request.image_base64)
    except binascii.Error:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"status": "error", "message": "Invalid base64 string"})

    try:
        predicted_pof, accuracy = pof(image_bytes,request.site_id,yolo_model, gnn_model)
        accuracy = accuracy * 100
        accuracy = round(accuracy, 2)

        try:
            image_path = save_dir / f'{request.order_id}.png'
            with open(image_path, 'wb') as file:
                file.write(image_bytes)
        except Exception as e:
            logger.error(f'Failed to save image for order id: {request.order_id}. Reason: {e}')
    except Exception as e:
        logger.error(f'Unexpected error occurred while processing order id: {request.order_id}. Reason: {e}')
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"status": "error", "message": "An unexpected error occurred on the server."})

    success_data = pofResponse(site_id=request.site_id, pof=predicted_pof, certainty=accuracy, order_id=request.order_id)
    return JSONResponse(status_code=status.HTTP_200_OK, content={
        "status": "success",
        "data": success_data.dict()
    })

@app.get('/health')
async def health_check(request: Request):
    return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "OK"})

if __name__ == '__main__':
    sys.exit(0)