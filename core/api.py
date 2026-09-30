"""
This contains api routes for the application. New routes may be added if need be
"""
import logging
logger = logging.getLogger(__name__)
logger.info('Loading modules...')

import sys
import base64
import binascii
from fastapi import FastAPI, status, Request, Depends
from typing import Annotated
from core.utils.schema import POFOUT, POFDTO, ErrorResponse, POFRESPONSE, HEALTHRESPONSE
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.body_limit import RequestBodyLimitMiddleware
from guard import SecurityMiddleware, SecurityConfig, SecurityDecorator
from core.middleware import RequestTimeoutMiddleware
from core.utils.pof import pof, prep_models
from core.utils.storage import save_received_image
from core.utils.helpers import is_demo_mode
from core.utils.exception_handler import InvalidImageException, SiteIdNotFoundInImage, NoSiteId
from core.utils.constants import SAVE_RECEIVED_IMAGES, MAX_BODY_SIZE_MB, REQUEST_TIMEOUT_S, ENFORCE_HTTPS, BEARER_TOKEN
from core.utils.auth import verify_bearer
from core.demo import load_demo_models
from contextlib import asynccontextmanager

logger.info('Modules loaded successfully')
if not BEARER_TOKEN:
    logger.warning('Bearer token not set, server will reject pof requests')

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

security_config = SecurityConfig(
    enable_redis=False,
    rate_limit=40,
    rate_limit_window=60,
    lazy_init=True,
    enable_rate_limiting=True,
    enable_penetration_detection=True,
    passive_mode=False,
    fail_secure=True,
    enforce_https=ENFORCE_HTTPS,
    enable_ip_banning=True,
    auto_ban_duration=1200,
    auto_ban_threshold=10,
    enable_rate_limit_auto_ban=True
)

guard = SecurityDecorator(config=security_config)

error_responses={
    400: {'model': ErrorResponse},
    401: {'model': ErrorResponse},
    500: {'model': ErrorResponse},
    503: {'model': ErrorResponse}
}

app.add_middleware(RequestTimeoutMiddleware, timeout_s=REQUEST_TIMEOUT_S)
app.add_middleware(RequestBodyLimitMiddleware, max_body_size=MAX_BODY_SIZE_MB * 1024 * 1024)
app.add_middleware(SecurityMiddleware, config=security_config)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    details = [{'type': err.get('type'), 'msg': err.get('msg')} for err in exc.errors() if isinstance(err, dict)]
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(message="Validation error", details=details).model_dump()
    )

@app.exception_handler(NoSiteId)
async def no_site_id_exception_handler(_: Request, exc: NoSiteId):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(message=exc.message).model_dump()
    )

@app.exception_handler(InvalidImageException)
async def image_exception_handler(_: Request, exc: InvalidImageException):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(message=exc.message).model_dump()
    )

@app.exception_handler(SiteIdNotFoundInImage)
async def site_id_exception_handler(_: Request, exc: SiteIdNotFoundInImage):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(message=exc.message).model_dump()
    )

@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(message='An unexpected error occurred, try again later').model_dump(),
    )

@app.post('/pof', response_model=POFRESPONSE, responses=error_responses)
def check_pof(body: POFDTO, http_request: Request, bearer_token: Annotated[str, Depends(verify_bearer)]):
    try:
        image_bytes = base64.b64decode(body.image_base64)
    except binascii.Error:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, 
            content=ErrorResponse(status='error', message="Invalid base64 string").model_dump()
        )

    try:
        models = http_request.app.state.models

        if models is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                content=ErrorResponse(message='Models not available, try again later').model_dump()
            )
                
        yolo_model, gnn_model = models
        
        predicted_pof, accuracy = pof(image_bytes,body.site_id,yolo_model, gnn_model)
        accuracy = accuracy * 100
        accuracy = round(accuracy, 2)

        if SAVE_RECEIVED_IMAGES:
            save_received_image(image_bytes, body.order_id)

    except (NoSiteId, InvalidImageException, SiteIdNotFoundInImage):
        raise
    except Exception as e:
        logger.error(f'Unexpected error occurred while processing order id: {body.order_id}. Reason: {e}')
        raise

    success_data = POFOUT(site_id=body.site_id, pof=predicted_pof, certainty=accuracy, order_id=body.order_id)
    return POFRESPONSE(
        status="success",
        data=success_data.model_dump()
    ).model_dump()

@app.get('/health', response_model=HEALTHRESPONSE)
async def health_check(request: Request):
    is_demo = is_demo_mode()
    environment = 'demo' if is_demo else 'live'
    app_status = 'healthy' if is_demo else ('healthy' if request.app.state.models else 'degraded')
    return HEALTHRESPONSE(status=app_status, environment=environment).model_dump()

if __name__ == '__main__':
    sys.exit(0)