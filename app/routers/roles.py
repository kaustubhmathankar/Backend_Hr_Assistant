from __future__ import annotations

from typing import Optional

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
    delete,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models import User
from app.rbac_models import (
    Permission,
    Role,
    role_permissions,
)
from app.services.rbac import (
    ALL_PERMISSION_KEYS,
    count_users_with_role,
    generate_role_key,
    has_permission,
    validate_permission_keys,
)


router = APIRouter(
    prefix="/admin/roles",
    tags=["Roles & Permissions"],
)


# ============================================================
# REQUEST SCHEMAS
# ============================================================


class RoleCreateRequest(
    BaseModel
):
    name: str = Field(
        ...,
        min_length=2,
        max_length=100,
    )

    description: Optional[str] = Field(
        default=None,
        max_length=1000,
    )

    permission_keys: list[str] = Field(
        default_factory=list,
    )


class RoleUpdateRequest(
    BaseModel
):
    name: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=100,
    )

    description: Optional[str] = Field(
        default=None,
        max_length=1000,
    )

    is_active: Optional[bool] = None

    permission_keys: Optional[
        list[str]
    ] = None


# ============================================================
# PERMISSION CHECK HELPER
# ============================================================


async def require_role_capability(
    *,
    current_user: User,
    permission_key: str,
    db: AsyncSession,
) -> None:

    allowed = await has_permission(
        current_user=current_user,
        permission_key=permission_key,
        db=db,
    )

    if not allowed:

        raise HTTPException(
            status_code=(
                status.HTTP_403_FORBIDDEN
            ),
            detail=(
                "Permission required: "
                f"{permission_key}"
            ),
        )


# ============================================================
# SERIALIZATION
# ============================================================


def serialize_permission(
    permission: Permission,
) -> dict:

    return {
        "id": permission.id,
        "key": permission.key,
        "name": permission.name,
        "module": permission.module,
        "description": permission.description,
    }


def serialize_role(
    role: Role,
    permission_keys: list[str],
    assigned_user_count: int,
) -> dict:

    return {
        "id": role.id,
        "key": role.key,
        "name": role.name,
        "description": role.description,
        "is_active": role.is_active,

        # Kept in the response for backward compatibility
        # with existing frontend code.
        # Current RBAC does not use system roles.
        "is_system": False,

        "permissions": sorted(
            permission_keys
        ),

        "assigned_user_count": (
            assigned_user_count
        ),

        "created_at": role.created_at,
        "updated_at": role.updated_at,
    }


# ============================================================
# LOAD ROLE PERMISSIONS
# ============================================================


async def load_role_permission_keys(
    role_id: int,
    db: AsyncSession,
) -> list[str]:

    result = await db.execute(
        select(
            Permission.key
        )
        .select_from(
            Permission
        )
        .join(
            role_permissions,
            role_permissions.c.permission_id
            == Permission.id,
        )
        .where(
            role_permissions.c.role_id
            == role_id
        )
        .order_by(
            Permission.key.asc()
        )
    )

    return list(
        result.scalars().all()
    )


# ============================================================
# GET PERMISSIONS
# ============================================================


@router.get(
    "/permissions"
)
async def list_permissions(
    current_user: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    await require_role_capability(
        current_user=current_user,
        permission_key="roles.view",
        db=db,
    )

    result = await db.execute(
        select(
            Permission
        )
        .order_by(
            Permission.module.asc(),
            Permission.name.asc(),
        )
    )

    permissions = (
        result.scalars().all()
    )

    return {
        "permissions": [
            serialize_permission(
                permission
            )
            for permission
            in permissions
        ],
        "total": len(
            permissions
        ),
        "all_permission_keys": sorted(
            ALL_PERMISSION_KEYS
        ),
    }


# ============================================================
# GET ALL ROLES
# ============================================================


@router.get("")
async def list_roles(
    current_user: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    await require_role_capability(
        current_user=current_user,
        permission_key="roles.view",
        db=db,
    )

    result = await db.execute(
        select(Role)
        .order_by(
            Role.name.asc()
        )
    )

    roles = (
        result.scalars().all()
    )

    response_roles = []

    for role in roles:

        permission_keys = (
            await load_role_permission_keys(
                role_id=role.id,
                db=db,
            )
        )

        assigned_user_count = (
            await count_users_with_role(
                db=db,
                role_key=role.key,
            )
        )

        response_roles.append(
            serialize_role(
                role=role,
                permission_keys=(
                    permission_keys
                ),
                assigned_user_count=(
                    assigned_user_count
                ),
            )
        )

    return {
        "roles": response_roles,
        "total": len(
            response_roles
        ),
    }


# ============================================================
# GET ONE ROLE
# ============================================================


@router.get(
    "/{role_key}"
)
async def get_role(
    role_key: str,
    current_user: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    await require_role_capability(
        current_user=current_user,
        permission_key="roles.view",
        db=db,
    )

    normalized_key = (
        role_key
        .strip()
        .lower()
    )

    result = await db.execute(
        select(Role).where(
            Role.key
            == normalized_key
        )
    )

    role = (
        result.scalar_one_or_none()
    )

    if role is None:

        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="Role not found.",
        )

    permission_keys = (
        await load_role_permission_keys(
            role_id=role.id,
            db=db,
        )
    )

    assigned_user_count = (
        await count_users_with_role(
            db=db,
            role_key=role.key,
        )
    )

    return {
        "role": serialize_role(
            role=role,
            permission_keys=(
                permission_keys
            ),
            assigned_user_count=(
                assigned_user_count
            ),
        )
    }


# ============================================================
# CREATE ROLE
# ============================================================


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
)
async def create_role(
    payload: RoleCreateRequest,
    current_user: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    await require_role_capability(
        current_user=current_user,
        permission_key="roles.create",
        db=db,
    )

    role_name = (
        payload.name
        .strip()
    )

    if not role_name:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Role name cannot be empty."
            ),
        )

    permission_keys = (
        await validate_permission_keys(
            payload.permission_keys
        )
    )

    role_key = generate_role_key(
        name=role_name,
        db=db,
    )

    # --------------------------------------------------------
    # Ensure key uniqueness.
    # --------------------------------------------------------

    while True:

        existing_result = await db.execute(
            select(Role.id).where(
                Role.key
                == role_key
            )
        )

        existing_role = (
            existing_result
            .scalar_one_or_none()
        )

        if existing_role is None:
            break

        role_key = generate_role_key(
            name=role_name,
            db=db,
        )

    role = Role(
        key=role_key,
        name=role_name,
        description=(
            payload.description.strip()
            if payload.description
            is not None
            else None
        ),
        is_system=False,
        is_active=True,
        created_by=current_user.id,
    )

    db.add(role)

    await db.flush()

    # --------------------------------------------------------
    # Attach permissions.
    # --------------------------------------------------------

    if permission_keys:

        permission_result = await db.execute(
            select(
                Permission
            ).where(
                Permission.key.in_(
                    permission_keys
                )
            )
        )

        permissions = {
            permission.key: permission
            for permission
            in permission_result
            .scalars()
            .all()
        }

        for permission_key in (
            permission_keys
        ):

            permission = (
                permissions.get(
                    permission_key
                )
            )

            if permission is None:

                raise HTTPException(
                    status_code=(
                        status.HTTP_400_BAD_REQUEST
                    ),
                    detail=(
                        "Permission not found: "
                        f"{permission_key}"
                    ),
                )

            # IMPORTANT:
            # role_permissions is a SQLAlchemy Core Table,
            # so INSERT must use db.execute(), not db.add().
            await db.execute(
                role_permissions
                .insert()
                .values(
                    role_id=role.id,
                    permission_id=(
                        permission.id
                    ),
                )
            )

    await db.commit()

    await db.refresh(
        role
    )

    return {
        "success": True,
        "message": (
            "Custom role created successfully."
        ),
        "role": serialize_role(
            role=role,
            permission_keys=(
                permission_keys
            ),
            assigned_user_count=0,
        ),
    }


# ============================================================
# UPDATE ROLE
# ============================================================


@router.put(
    "/{role_key}"
)
async def update_role(
    role_key: str,
    payload: RoleUpdateRequest,
    current_user: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    normalized_key = (
        role_key
        .strip()
        .lower()
    )

    result = await db.execute(
        select(Role).where(
            Role.key
            == normalized_key
        )
    )

    role = (
        result.scalar_one_or_none()
    )

    if role is None:

        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="Role not found.",
        )

    changing_metadata = any(
        value is not None
        for value in (
            payload.name,
            payload.description,
            payload.is_active,
        )
    )

    changing_permissions = (
        payload.permission_keys
        is not None
    )

    if (
        not changing_metadata
        and not changing_permissions
    ):

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "At least one role field "
                "must be provided."
            ),
        )

    # --------------------------------------------------------
    # Prevent a normal user from modifying the role they
    # currently use.
    #
    # Super Admin remains protected by:
    # current_user.role == "admin"
    # --------------------------------------------------------

    if (
        current_user.role == role.key
        and current_user.role != "admin"
    ):

        raise HTTPException(
            status_code=(
                status.HTTP_403_FORBIDDEN
            ),
            detail=(
                "You cannot modify the role "
                "currently assigned to you."
            ),
        )

    # --------------------------------------------------------
    # Role metadata.
    # --------------------------------------------------------

    if changing_metadata:

        await require_role_capability(
            current_user=current_user,
            permission_key="roles.update",
            db=db,
        )

    # --------------------------------------------------------
    # Role permissions.
    # --------------------------------------------------------

    permission_keys = None

    if changing_permissions:

        await require_role_capability(
            current_user=current_user,
            permission_key=(
                "roles.manage_permissions"
            ),
            db=db,
        )

        permission_keys = (
            await validate_permission_keys(
                payload.permission_keys
                or []
            )
        )

        # Remove old permission associations.
        await db.execute(
            delete(
                role_permissions
            ).where(
                role_permissions.c.role_id
                == role.id
            )
        )

        if permission_keys:

            permission_result = await db.execute(
                select(
                    Permission
                ).where(
                    Permission.key.in_(
                        permission_keys
                    )
                )
            )

            permissions = {
                permission.key: permission
                for permission
                in permission_result
                .scalars()
                .all()
            }

            for permission_key in (
                permission_keys
            ):

                permission = (
                    permissions.get(
                        permission_key
                    )
                )

                if permission is None:

                    raise HTTPException(
                        status_code=(
                            status.HTTP_400_BAD_REQUEST
                        ),
                        detail=(
                            "Permission not found: "
                            f"{permission_key}"
                        ),
                    )

                # IMPORTANT:
                # role_permissions is a SQLAlchemy Core Table,
                # so INSERT must use db.execute(), not db.add().
                await db.execute(
                    role_permissions
                    .insert()
                    .values(
                        role_id=role.id,
                        permission_id=(
                            permission.id
                        ),
                    )
                )

    # --------------------------------------------------------
    # Update name.
    # --------------------------------------------------------

    if payload.name is not None:

        role_name = (
            payload.name
            .strip()
        )

        if not role_name:

            raise HTTPException(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                ),
                detail=(
                    "Role name cannot be empty."
                ),
            )

        role.name = role_name

    # --------------------------------------------------------
    # Update description.
    # --------------------------------------------------------

    if payload.description is not None:

        role.description = (
            payload.description
            .strip()
        )

    # --------------------------------------------------------
    # Activate/deactivate role.
    # --------------------------------------------------------

    if payload.is_active is not None:

        if not payload.is_active:

            assigned_user_count = (
                await count_users_with_role(
                    db=db,
                    role_key=role.key,
                )
            )

            if assigned_user_count > 0:

                raise HTTPException(
                    status_code=(
                        status.HTTP_409_CONFLICT
                    ),
                    detail=(
                        "This role is currently "
                        "assigned to one or more users. "
                        "Reassign those users before "
                        "deactivating the role."
                    ),
                )

        role.is_active = (
            payload.is_active
        )

    await db.commit()

    await db.refresh(
        role
    )

    final_permission_keys = (
        await load_role_permission_keys(
            role_id=role.id,
            db=db,
        )
    )

    assigned_user_count = (
        await count_users_with_role(
            db=db,
            role_key=role.key,
        )
    )

    return {
        "success": True,
        "message": (
            "Role updated successfully."
        ),
        "role": serialize_role(
            role=role,
            permission_keys=(
                final_permission_keys
            ),
            assigned_user_count=(
                assigned_user_count
            ),
        ),
    }


# ============================================================
# DELETE ROLE
# ============================================================


@router.delete(
    "/{role_key}"
)
async def delete_role(
    role_key: str,
    current_user: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    await require_role_capability(
        current_user=current_user,
        permission_key="roles.delete",
        db=db,
    )

    normalized_key = (
        role_key
        .strip()
        .lower()
    )

    result = await db.execute(
        select(Role).where(
            Role.key
            == normalized_key
        )
    )

    role = (
        result.scalar_one_or_none()
    )

    if role is None:

        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="Role not found.",
        )

    # --------------------------------------------------------
    # Every Role is a custom dynamically created role.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Prevent deleting the role currently assigned to the
    # requesting user.
    # --------------------------------------------------------

    if current_user.role == role.key:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "You cannot delete the role "
                "currently assigned to you."
            ),
        )

    # --------------------------------------------------------
    # Assigned roles cannot be deleted.
    # --------------------------------------------------------

    assigned_user_count = (
        await count_users_with_role(
            db=db,
            role_key=role.key,
        )
    )

    if assigned_user_count > 0:

        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
            ),
            detail=(
                "This role cannot be deleted "
                "while users are assigned to it."
            ),
        )

    # --------------------------------------------------------
    # Remove permission associations.
    # --------------------------------------------------------

    await db.execute(
        delete(
            role_permissions
        ).where(
            role_permissions.c.role_id
            == role.id
        )
    )

    await db.delete(
        role
    )

    await db.commit()

    return {
        "success": True,
        "message": (
            "Role deleted successfully."
        ),
        "role_key": normalized_key,
    }