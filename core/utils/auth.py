import secrets
from fastapi import Header, HTTPException, status
from core.utils.constants import BEARER_TOKEN

def verify_bearer(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.strip():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Auth Token")

    if not BEARER_TOKEN:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Server auth not configured")

    parts = authorization.strip().split(None, 1)
    if len(parts) != 2:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Bearer Token")

    scheme, token = parts
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Bearer Token")

    if not secrets.compare_digest(token, BEARER_TOKEN):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Bearer Token")
    return token