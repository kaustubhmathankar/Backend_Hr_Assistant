from fastapi import APIRouter, Depends

from app.auth import get_current_user, require_user
from app.models import User


router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


# ============================================================
# GET CURRENT LOGGED-IN USER
# ============================================================

@router.get("/me")
async def get_my_profile(
    current_user: User = Depends(get_current_user),
):
    return {
        "message": "Authenticated successfully.",
        "user": {
            "id": current_user.id,
            "email": current_user.email,
            "role": current_user.role,
            "is_active": current_user.is_active,
        },
    }


# ============================================================
# NORMAL USER ONLY
# ============================================================

@router.get("/dashboard")
async def user_dashboard(
    current_user: User = Depends(require_user),
):
    return {
        "message": "Welcome to the user dashboard.",
        "user_id": current_user.id,
        "email": current_user.email,
        "role": current_user.role,
    }