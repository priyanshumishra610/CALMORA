"""
Exception handling utilities for Calmora backend.
"""
import logging
from fastapi import HTTPException

def handle_exception(e: Exception, context: str = ""):
    logging.error(f"Exception in {context}: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}") 