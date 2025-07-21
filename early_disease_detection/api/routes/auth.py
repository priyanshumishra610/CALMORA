"""
Auth and Legal Endpoints for Calmora
"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/consent", tags=["legal"])
def get_consent_policy():
    """
    Returns the Calmora consent/legal policy text.
    """
    return {
        "policy": (
            "Calmora is an AI health companion. It does not provide medical advice, diagnosis, or treatment. "
            "All information is for informational purposes only. Always consult a qualified healthcare provider for medical concerns. "
            "By using this service, you acknowledge and accept this policy."
        )
    } 