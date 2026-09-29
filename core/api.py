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
from starlette.middleware.body_limit import RequestBodyLimitMiddleware
from core.utils.pof import pof, prep_models
from core.utils.storage import save_received_image
from core.utils.helpers import is_demo_mode
from core.utils.exception_handler import InvalidImageException, SiteIdNotFoundInImage, NoSiteId
from core.utils.constants import SAVE_RECEIVED_IMAGES, MAX_BODY_SIZE_MB
from core.demo import load_demo_models
from contextlib import asynccontextmanager

logger.info('Modules loaded successfully')

@asynccontextmanager
async def lifespan(app: FastAPI):
    yolo_model_path = 'models/YOLO/best.pt'
    gnn_model_path = 'models/GNN/best.pt'

    if is_demo_mode():
        app.state.models = load_demo_models()
    else:
        app.state.models = prep_models(yolo_model_path, gnn_model_path)

    yield

    logger.info('Shutting down server...')

app = FastAPI(lifespan=lifespan)
app.add_middleware(RequestBodyLimitMiddleware, max_body_size=MAX_BODY_SIZE_MB * 1024 * 1024)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    details = [{'type': err.get('type'), 'msg': err.get('msg')} for err in exc.errors() if isinstance(err, dict)]
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"status": "error", "message": "Validation error", "details" : details}
    )

@app.exception_handler(NoSiteId)
async def no_site_id_exception_handler(_: Request, exc: NoSiteId):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"status": "error", "message": exc.message}
    )

@app.exception_handler(InvalidImageException)
async def image_exception_handler(_: Request, exc: InvalidImageException):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"status": "error", "message": exc.message}
    )

@app.exception_handler(SiteIdNotFoundInImage)
async def site_id_exception_handler(_: Request, exc: SiteIdNotFoundInImage):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"status": "error", "message" : exc.message}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "message": "An unexpected error occurred, try again later"},
    )

@app.post('/pof')
async def check_pof(body: pofRequest, http_request: Request):
    try:
        image_bytes = base64.b64decode(body.image_base64)
    except binascii.Error:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"status": "error", "message": "Invalid base64 string"})

    try:
        models = http_request.app.state.models

        if models is None:
            return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={'status': 'error', 'message': 'Models not available, try again later'})
                
        yolo_model, gnn_model = models
        
        predicted_pof, accuracy = pof(image_bytes,body.site_id,yolo_model, gnn_model)
        accuracy = accuracy * 100
        accuracy = round(accuracy, 2)

        #chore: sanitize this image
        if SAVE_RECEIVED_IMAGES:
            save_received_image(image_bytes, body.order_id)

    except (NoSiteId, InvalidImageException, SiteIdNotFoundInImage):
        raise
    except Exception as e:
        logger.error(f'Unexpected error occurred while processing order id: {body.order_id}. Reason: {e}')
        raise

    success_data = pofResponse(site_id=body.site_id, pof=predicted_pof, certainty=accuracy, order_id=body.order_id)
    return JSONResponse(status_code=status.HTTP_200_OK, content={
        "status": "success",
        "data": success_data.model_dump()
    })

@app.get('/health')
async def health_check(request: Request):
    is_demo = is_demo_mode()
    environment = 'demo' if is_demo else 'live'
    app_status = 'healthy' if request.app.state.models is not None else 'degraded'
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": app_status, 'environment': environment})

if __name__ == '__main__':
    sys.exit(0)