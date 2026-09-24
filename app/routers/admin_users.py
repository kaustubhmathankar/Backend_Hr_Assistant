from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from pydantic import (
    BaseModel,
    Field,
)

from sqlalchemy import (
    func,
    select,
)

from sqlalchemy.ext.asyncio import (
    AsyncSession,
)

from app.auth import (
    get_current_user,
)

from app.database import (
    get_db,
)

from app.models import (
    User,
)

from app.rbac_models import (
    Role,
)

from app.services.rbac import (
    require_permission,
    has_permission,
)


router = APIRouter(
    prefix="/admin/users",
    tags=["Admin Users"],
)


# ============================================================
# REQUEST MODELS
# ============================================================


class UserRoleUpdateRequest(
    BaseModel
):
    role: str = Field(
        min_length=1,
        max_length=20,
    )


class UserStatusUpdateRequest(
    BaseModel
):
    is_active: bool


# ============================================================
# HELPERS
# ============================================================


def serialize_user(
    user: User,
) -> dict:

    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "is_active": user.is_active,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
    }


async def get_target_user(
    user_id: int,
    db: AsyncSession,
) -> User:

    result = await db.execute(
        select(User).where(
            User.id == user_id
        )
    )

    user = (
        result.scalar_one_or_none()
    )

    if user is None:

        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="User not found.",
        )

    return user


async def count_active_admins(
    db: AsyncSession,
) -> int:

    result = await db.execute(
        select(
            func.count(User.id)
        ).where(
            User.role == "admin",
            User.is_active.is_(True),
        )
    )

    return (
        result.scalar()
        or 0
    )


async def validate_assignable_role(
    role_key: str,
    db: AsyncSession,
) -> str:

    normalized_role = (
        role_key
        .strip()
        .lower()
    )

    # --------------------------------------------------------
    # Legacy normal user role.
    # --------------------------------------------------------

    if normalized_role == "user":
        return normalized_role


    # --------------------------------------------------------
    # Reserved Super Admin role.
    #
    # Only the real Super Admin can assign this role.
    # This check is performed separately by the endpoint.
    # --------------------------------------------------------

    if normalized_role == "admin":
        return normalized_role


    # --------------------------------------------------------
    # Custom role.
    # --------------------------------------------------------

    result = await db.execute(
        select(Role).where(
            Role.key == normalized_role,
            Role.is_active.is_(True),
        )
    )

    role = (
        result.scalar_one_or_none()
    )

    if role is None:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                f"Active role '{normalized_role}' "
                "was not found."
            ),
        )

    return normalized_role


# ============================================================
# LIST USERS
# ============================================================


@router.get("")
async def list_admin_users(
    current_user: User = Depends(
        require_permission("users.view")
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    # current_user is already validated by the dependency
    result = await db.execute(
        select(User)
        .order_by(
            User.created_at.desc()
        )
    )

    users = (
        result.scalars().all()
    )

    return {
        "users": [
            serialize_user(user)
            for user in users
        ],
        "total": len(users),
    }


# ============================================================
# GET ONE USER
# ============================================================


@router.get(
    "/{user_id}"
)
async def get_admin_user(
    user_id: int,
    current_user: User = Depends(
        require_permission("users.view")
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    # current_user is already validated by the dependency
    user = await get_target_user(
        user_id=user_id,
        db=db,
    )

    return {
        "user": serialize_user(user)
    }


# ============================================================
# UPDATE USER ROLE
# ============================================================


@router.put(
    "/{user_id}/role"
)
async def update_user_role(
    user_id: int,
    payload: UserRoleUpdateRequest,
    current_user: User = Depends(
        require_permission("users.assign_role")
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    # current_user is already validated by the dependency


    target_user = await get_target_user(
        user_id=user_id,
        db=db,
    )


    new_role = await validate_assignable_role(
        role_key=payload.role,
        db=db,
    )


    # --------------------------------------------------------
    # Prevent changing your own role.
    # --------------------------------------------------------

    if (
        target_user.id ==
        current_user.id
    ):

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "You cannot change your own role."
            ),
        )


    # --------------------------------------------------------
    # Only Super Admin can assign the reserved
    # Super Admin role.
    # --------------------------------------------------------

    if (
        new_role == "admin"
        and current_user.role != "admin"
    ):

        raise HTTPException(
            status_code=(
                status.HTTP_403_FORBIDDEN
            ),
            detail=(
                "Only the Super Admin can assign "
                "the Super Admin role."
            ),
        )


    # --------------------------------------------------------
    # No-op.
    # --------------------------------------------------------

    if (
        target_user.role ==
        new_role
    ):

        return {
            "success": True,
            "message": (
                "User role is already "
                f"'{new_role}'."
            ),
            "user": serialize_user(
                target_user
            ),
        }


    # --------------------------------------------------------
    # Last Super Admin protection.
    #
    # The last active Super Admin cannot be moved
    # to any other role.
    # --------------------------------------------------------

    if (
        target_user.role == "admin"
        and new_role != "admin"
        and target_user.is_active
    ):

        active_admin_count = (
            await count_active_admins(
                db
            )
        )

        if (
            active_admin_count <= 1
        ):

            raise HTTPException(
                status_code=(
                    status.HTTP_409_CONFLICT
                ),
                detail=(
                    "The last active "
                    "administrator cannot "
                    "be demoted."
                ),
            )


    target_user.role = new_role


    await db.commit()

    await db.refresh(
        target_user
    )


    return {
        "success": True,
        "message": (
            "User role updated successfully."
        ),
        "user": serialize_user(
            target_user
        ),
    }


# ============================================================
# UPDATE USER STATUS
# ============================================================


@router.put(
    "/{user_id}/status"
)
async def update_user_status(
    user_id: int,
    payload: UserStatusUpdateRequest,
    current_user: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    new_status = bool(
        payload.is_active
    )


    # --------------------------------------------------------
    # Permission depends on the requested action.
    # --------------------------------------------------------

    required_permission = (
        "users.activate"
        if new_status
        else "users.deactivate"
    )


    # Programmatic permission check (required_permission depends on payload)
    allowed = await has_permission(
        current_user,
        required_permission,
        db,
    )

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Permission required: "
                f"{required_permission}"
            ),
        )


    target_user = await get_target_user(
        user_id=user_id,
        db=db,
    )


    # --------------------------------------------------------
    # No-op.
    # --------------------------------------------------------

    if (
        target_user.is_active
        == new_status
    ):

        return {
            "success": True,
            "message": (
                "User status is already "
                f"{'active' if new_status else 'inactive'}."
            ),
            "user": serialize_user(
                target_user
            ),
        }


    # --------------------------------------------------------
    # Prevent self-deactivation.
    # --------------------------------------------------------

    if (
        target_user.id ==
        current_user.id
        and not new_status
    ):

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "You cannot deactivate "
                "your own account."
            ),
        )


    # --------------------------------------------------------
    # Prevent deactivating the last active
    # Super Admin.
    # --------------------------------------------------------

    if (
        target_user.role == "admin"
        and target_user.is_active
        and not new_status
    ):

        active_admin_count = (
            await count_active_admins(
                db
            )
        )

        if (
            active_admin_count <= 1
        ):

            raise HTTPException(
                status_code=(
                    status.HTTP_409_CONFLICT
                ),
                detail=(
                    "The last active "
                    "administrator cannot "
                    "be deactivated."
                ),
            )


    target_user.is_active = new_status


    await db.commit()

    await db.refresh(
        target_user
    )


    return {
        "success": True,
        "message": (
            "User status updated successfully."
        ),
        "user": serialize_user(
            target_user
        ),
    }