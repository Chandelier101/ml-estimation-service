"""
Dependencies module for FastAPI app to manage request-level dependencies, such as authentication.
"""

from fastapi import Header, HTTPException

async def verify_token(x_token: str = Header(...)):
    """
    Verifies the provided token against an expected value.

    Args:
        x_token (str): Token from the request header.

    Raises:
        HTTPException: If the token does not match the expected value.
    """
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")
