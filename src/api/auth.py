# auth.py
# API key authentication dependency for FastAPI.
# Any route that includes Depends(verify_api_key) will require a valid X-API-Key header before the route handler runs.

# How FastAPI dependencies work:
# When a request hits a protected endpoint, FastAPI calls verify_api_key() first. If it raises HTTPException, the request stops there and the client gets a 401 response. If it passes, the route handler runs normally.

import os
from fastapi import Header, HTTPException
from dotenv import load_dotenv

load_dotenv()


async def verify_api_key(x_api_key: str = Header(...)):
    """
    FastAPI dependency that validates the X-API-Key request header.

    Header(...) tells FastAPI to extract the X-API-Key header from the incoming request automatically. The '...' means it is required; if the header is missing entirely, FastAPI returns 422 before this function even runs.

    We then compare the provided key against the value stored in .env.
    If they do not match, we return 401 Unauthorized.
    If they match, the function returns normally and the route runs.
    """
    expected_key = os.getenv("API_KEY")

    if not expected_key:
        # This means API_KEY is missing from .env entirely
        # We fail closed: deny all requests rather than accidentally leaving the API open due to a misconfiguration
        raise HTTPException(
            status_code=500,
            detail="API key not configured on server."
        )

    if x_api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key."
        )