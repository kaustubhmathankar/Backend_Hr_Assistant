from __future__ import annotations

import re
from uuid import uuid4

from fastapi import (
    Depends,
    HTTPException,
    status,
)
from sqlalchemy import (
    func,
    select,
    update,
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


# ============================================================
# PERMISSION CATALOG
# ============================================================
#
# Permissions are application capabilities.
#
# Roles are NOT hardcoded here.
#
# The Super Admin creates roles dynamically and selects any
# combination of these permissions for each role.
# ============================================================

PERMISSION_DEFINITIONS = [
    {
        "key": "dashboard.view",
        "name": "View Dashboard",
        "module": "Dashboard",
        "description": (
            "View the application dashboard."
        ),
    },

    # --------------------------------------------------------
    # USERS
    # --------------------------------------------------------

    {
        "key": "users.view",
        "name": "View Users",
        "module": "Users",
        "description": (
            "View user accounts."
        ),
    },
    {
        "key": "users.assign_role",
        "name": "Assign Roles",
        "module": "Users",
        "description": (
            "Assign roles to users."
        ),
    },
    {
        "key": "users.activate",
        "name": "Activate Users",
        "module": "Users",
        "description": (
            "Activate user accounts."
        ),
    },
    {
        "key": "users.deactivate",
        "name": "Deactivate Users",
        "module": "Users",
        "description": (
            "Deactivate user accounts."
        ),
    },

    # --------------------------------------------------------
    # ROLES
    # --------------------------------------------------------

    {
        "key": "roles.view",
        "name": "View Roles",
        "module": "Roles",
        "description": (
            "View roles and their permissions."
        ),
    },
    {
        "key": "roles.create",
        "name": "Create Roles",
        "module": "Roles",
        "description": (
            "Create custom roles."
        ),
    },
    {
        "key": "roles.update",
        "name": "Update Roles",
        "module": "Roles",
        "description": (
            "Update role metadata."
        ),
    },
    {
        "key": "roles.delete",
        "name": "Delete Roles",
        "module": "Roles",
        "description": (
            "Delete roles."
        ),
    },
    {
        "key": "roles.manage_permissions",
        "name": "Manage Role Permissions",
        "module": "Roles",
        "description": (
            "Change permissions assigned to a role."
        ),
    },

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    {
        "key": "ai.providers.view",
        "name": "View AI Providers",
        "module": "AI",
        "description": (
            "View configured AI providers."
        ),
    },
    {
        "key": "ai.providers.manage",
        "name": "Manage AI Providers",
        "module": "AI",
        "description": (
            "Manage AI provider configuration."
        ),
    },
    {
        "key": "ai.credentials.manage",
        "name": "Manage AI Credentials",
        "module": "AI",
        "description": (
            "Manage provider API credentials."
        ),
    },
    {
        "key": "ai.models.view",
        "name": "View AI Models",
        "module": "AI",
        "description": (
            "View available provider models."
        ),
    },
    {
        "key": "ai.runtime.view",
        "name": "View AI Runtime",
        "module": "AI",
        "description": (
            "View AI runtime configuration."
        ),
    },
    {
        "key": "ai.runtime.manage",
        "name": "Manage AI Runtime",
        "module": "AI",
        "description": (
            "Change AI runtime configuration."
        ),
    },

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    {
        "key": "rag.settings.view",
        "name": "View RAG Settings",
        "module": "RAG",
        "description": (
            "View RAG configuration."
        ),
    },
    {
        "key": "rag.settings.manage",
        "name": "Manage RAG Settings",
        "module": "RAG",
        "description": (
            "Change RAG configuration."
        ),
    },
    {
        "key": "rag.index.view",
        "name": "View RAG Index",
        "module": "RAG",
        "description": (
            "View RAG index information."
        ),
    },
    {
        "key": "rag.index.rebuild",
        "name": "Rebuild RAG Index",
        "module": "RAG",
        "description": (
            "Rebuild the RAG index."
        ),
    },

    # --------------------------------------------------------
    # DOCUMENTS
    # --------------------------------------------------------

    {
        "key": "documents.own.view",
        "name": "View Own Documents",
        "module": "Documents",
        "description": (
            "View documents owned by the current user."
        ),
    },
    {
        "key": "documents.own.upload",
        "name": "Upload Own Documents",
        "module": "Documents",
        "description": (
            "Upload documents for the current user."
        ),
    },
    {
        "key": "documents.own.delete",
        "name": "Delete Own Documents",
        "module": "Documents",
        "description": (
            "Delete documents owned by the current user."
        ),
    },
    {
        "key": "documents.all.view",
        "name": "View All Documents",
        "module": "Documents",
        "description": (
            "View documents across users."
        ),
    },
    {
        "key": "documents.all.upload",
        "name": "Upload Documents for All",
        "module": "Documents",
        "description": (
            "Upload documents on behalf of the organization."
        ),
    },
    {
        "key": "documents.all.delete",
        "name": "Delete All Documents",
        "module": "Documents",
        "description": (
            "Delete documents across users."
        ),
    },

    # --------------------------------------------------------
    # CHAT
    # --------------------------------------------------------

    {
        "key": "chat.use",
        "name": "Use Chat",
        "module": "Chat",
        "description": (
            "Use the HR Assistant."
        ),
    },

    # --------------------------------------------------------
    # CONVERSATIONS
    # --------------------------------------------------------

    {
        "key": "conversations.own.view",
        "name": "View Own Conversations",
        "module": "Conversations",
        "description": (
            "View the current user's conversations."
        ),
    },
    {
        "key": "conversations.own.delete",
        "name": "Delete Own Conversations",
        "module": "Conversations",
        "description": (
            "Delete the current user's conversations."
        ),
    },
    {
        "key": "conversations.all.view",
        "name": "View All Conversations",
        "module": "Conversations",
        "description": (
            "View conversations across users."
        ),
    },
    {
        "key": "conversations.all.delete",
        "name": "Delete All Conversations",
        "module": "Conversations",
        "description": (
            "Delete conversations across users."
        ),
    },
]


ALL_PERMISSION_KEYS = {
    permission["key"]
    for permission in PERMISSION_DEFINITIONS
}


# ============================================================
# NORMALIZE PERMISSION
# ============================================================

def normalize_permission_key(
    permission_key: str,
) -> str:
    return (
        permission_key
        .strip()
        .lower()
    )


# ============================================================
# GET ROLE
# ============================================================

async def get_role_by_key(
    db: AsyncSession,
    role_key: str,
) -> Role | None:

    normalized_key = (
        role_key
        .strip()
        .lower()
    )

    result = await db.execute(
        select(Role).where(
            Role.key == normalized_key,
            Role.is_active.is_(True),
        )
    )

    return result.scalar_one_or_none()


# ============================================================
# GET PERMISSION
# ============================================================

async def get_permission_by_key(
    db: AsyncSession,
    permission_key: str,
) -> Permission | None:

    normalized_key = (
        normalize_permission_key(
            permission_key
        )
    )

    result = await db.execute(
        select(Permission).where(
            Permission.key
            == normalized_key
        )
    )

    return result.scalar_one_or_none()


# ============================================================
# GET USER PERMISSIONS
# ============================================================

async def get_user_permission_keys(
    current_user: User,
    db: AsyncSession,
) -> set[str]:

    # --------------------------------------------------------
    # Existing Super Admin compatibility.
    #
    # The "admin" User.role is the protected bootstrap
    # Super Admin account. It is NOT a configurable Role.
    # --------------------------------------------------------

    if current_user.role == "admin":

        return set(
            ALL_PERMISSION_KEYS
        )

    # --------------------------------------------------------
    # Normal users obtain permissions exclusively through the
    # dynamically assigned role.
    # --------------------------------------------------------

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
        .join(
            Role,
            Role.id
            == role_permissions.c.role_id,
        )
        .where(
            Role.key
            == current_user.role,
            Role.is_active.is_(True),
        )
    )

    permissions = set(result.scalars().all())

    # Allow chat access for all roles by default (not restricted).
    # This ensures every role can use the chat feature unless the
    # Super Admin explicitly removes the permission from a role.
    try:
        permissions.add(normalize_permission_key("chat.use"))
    except Exception:
        # In case normalize_permission_key changes or fails for any reason,
        # fall back to adding the literal key.
        permissions.add("chat.use")

    return permissions


# ============================================================
# CHECK PERMISSION
# ============================================================

async def has_permission(
    current_user: User,
    permission_key: str,
    db: AsyncSession,
) -> bool:

    normalized_permission = (
        normalize_permission_key(
            permission_key
        )
    )

    permissions = (
        await get_user_permission_keys(
            current_user=current_user,
            db=db,
        )
    )

    return (
        normalized_permission
        in permissions
    )


# ============================================================
# REQUIRE PERMISSION
# ============================================================

def require_permission(
    permission_key: str,
):

    normalized_permission = (
        normalize_permission_key(
            permission_key
        )
    )

    async def dependency(
        current_user: User = Depends(
            get_current_user
        ),
        db: AsyncSession = Depends(
            get_db
        ),
    ) -> User:

        allowed = await has_permission(
            current_user=current_user,
            permission_key=(
                normalized_permission
            ),
            db=db,
        )

        if not allowed:

            raise HTTPException(
                status_code=(
                    status.HTTP_403_FORBIDDEN
                ),
                detail=(
                    "Permission required: "
                    f"{normalized_permission}"
                ),
            )

        return current_user

    return dependency


# ============================================================
# REQUIRE ANY PERMISSION
# ============================================================

def require_any_permission(
    *permission_keys: str,
):

    normalized_permissions = {
        normalize_permission_key(
            permission_key
        )
        for permission_key
        in permission_keys
    }

    async def dependency(
        current_user: User = Depends(
            get_current_user
        ),
        db: AsyncSession = Depends(
            get_db
        ),
    ) -> User:

        granted = (
            await get_user_permission_keys(
                current_user=current_user,
                db=db,
            )
        )

        if not normalized_permissions.intersection(
            granted
        ):

            raise HTTPException(
                status_code=(
                    status.HTTP_403_FORBIDDEN
                ),
                detail=(
                    "One of the required "
                    "permissions is missing."
                ),
            )

        return current_user

    return dependency


# ============================================================
# VALIDATE PERMISSION KEYS
# ============================================================

async def validate_permission_keys(
    permission_keys: list[str],
) -> list[str]:

    normalized = []
    seen = set()

    for permission_key in (
        permission_keys
    ):

        normalized_key = (
            normalize_permission_key(
                permission_key
            )
        )

        if not normalized_key:
            continue

        if normalized_key not in (
            ALL_PERMISSION_KEYS
        ):

            raise HTTPException(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                ),
                detail=(
                    "Unknown permission: "
                    f"{normalized_key}"
                ),
            )

        if normalized_key in seen:
            continue

        normalized.append(
            normalized_key
        )

        seen.add(
            normalized_key
        )

    return normalized


# ============================================================
# GENERATE ROLE KEY
# ============================================================
#
# Existing users.role is limited to 20 characters.
#
# Therefore:
#
#     visible name:
#         "Senior Recruitment Manager"
#
# can have an internal key such as:
#
#     senior-recru-a91c3f
#
# The key is only an internal database identifier.
# ============================================================

def generate_role_key(
    name: str,
    db: AsyncSession | None = None,
) -> str:

    slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        name.strip().lower(),
    )

    slug = slug.strip("-")

    if not slug:
        slug = "custom-role"

    base = slug[:12]

    return (
        f"{base}-"
        f"{uuid4().hex[:6]}"
    )


# ============================================================
# INITIALIZE RBAC
# ============================================================
#
# IMPORTANT:
#
# This function creates ONLY permissions.
#
# It DOES NOT create roles.
#
# Every role is created dynamically by the Super Admin.
# ============================================================

async def initialize_rbac(
    db: AsyncSession,
) -> None:

    # --------------------------------------------------------
    # Ensure all permission definitions exist.
    # --------------------------------------------------------

    existing_result = await db.execute(
        select(
            Permission
        )
    )

    existing_permissions = {
        permission.key: permission
        for permission
        in existing_result.scalars().all()
    }

    for definition in (
        PERMISSION_DEFINITIONS
    ):

        permission_key = (
            definition["key"]
        )

        if permission_key in (
            existing_permissions
        ):
            continue

        db.add(
            Permission(
                key=permission_key,
                name=definition["name"],
                module=definition["module"],
                description=definition[
                    "description"
                ],
            )
        )

    # --------------------------------------------------------
    # Make sure newly-added permissions are visible during
    # the same initialization transaction.
    # --------------------------------------------------------

    await db.flush()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Older versions of the RBAC implementation used
    # "system roles".
    #
    # Current architecture does NOT use system roles.
    #
    # Existing role records are therefore converted into
    # normal custom roles.
    # --------------------------------------------------------

    await db.execute(
        update(Role).values(
            is_system=False
        )
    )

    await db.commit()


# ============================================================
# COUNT USERS WITH ROLE
# ============================================================

async def count_users_with_role(
    db: AsyncSession,
    role_key: str,
) -> int:

    normalized_key = (
        role_key
        .strip()
        .lower()
    )

    result = await db.execute(
        select(
            func.count(
                User.id
            )
        ).where(
            User.role
            == normalized_key
        )
    )

    return int(
        result.scalar() or 0
    )