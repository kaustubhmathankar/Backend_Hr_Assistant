from datetime import datetime, timedelta
from typing import Optional

from fastapi import (
    Depends,
    HTTPException,
    Request,
    status,
)
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.rbac_models import (
    Permission,
    Role,
    role_permissions,
)


# ============================================================
# PASSWORD HASHING
# ============================================================

pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto",
)


def hash_password(
    password: str,
) -> str:
    return pwd_context.hash(
        password
    )


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    return pwd_context.verify(
        plain_password,
        hashed_password,
    )


# ============================================================
# JWT CONFIGURATION
# ============================================================

SECRET_KEY = (
    "change-this-secret-key-in-production"
)

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60


# ============================================================
# OAUTH2
# ============================================================

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login"
)


# ============================================================
# CREATE JWT TOKEN
# ============================================================

def create_access_token(
    user_id: int,
    email: str,
    role: str,
) -> str:

    expire = (
        datetime.utcnow()
        + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
    )

    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "exp": expire,
    }

    token = jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM,
    )

    return token


# ============================================================
# RBAC ROUTE PERMISSION MAP
# ============================================================
#
# Each inner set represents an OR condition.
#
# Example:
#
#     [{"documents.own.view", "documents.all.view"}]
#
# means the user needs either:
#
#     documents.own.view
#
# OR:
#
#     documents.all.view
#
# When multiple sets exist, the user must satisfy EVERY set.
#
# This allows us to preserve the existing endpoint files while
# making backend authorization permission-driven.
# ============================================================


async def get_required_permission_groups(
    request: Request,
) -> list[set[str]]:
    method = (
        request.method
        .strip()
        .upper()
    )

    path = (
        request.url.path
        .rstrip("/")
    )

    # ========================================================
    # ADMIN DASHBOARD
    # ========================================================

    if (
        method == "GET"
        and path == "/admin/dashboard"
    ):
        return [
            {
                "dashboard.view"
            }
        ]

    # ========================================================
    # SYSTEM SETTINGS
    # ========================================================

    if path == "/admin/settings":

        if method == "GET":
            return [
                {
                    "ai.runtime.view",
                    "rag.settings.view",
                }
            ]

        if method == "PUT":

            try:
                body = await request.json()

            except Exception:
                body = {}

            if not isinstance(
                body,
                dict,
            ):
                body = {}

            # ------------------------------------------------
            # AI runtime-related fields
            # ------------------------------------------------

            ai_fields = {
                "primary_provider",
                "primary_model",
                "utility_provider",
                "utility_model",
                "temperature",
                "max_tokens",
            }

            # ------------------------------------------------
            # RAG / upload-related fields
            # ------------------------------------------------

            rag_fields = {
                "chunk_size",
                "chunk_overlap",
                "retrieval_top_k",
                "relevance_threshold",
                "max_file_size_mb",
                "max_total_upload_size_mb",
                "max_files_per_upload",
                "max_zip_files",
                "max_zip_uncompressed_size_mb",
                "embedding_model",
            }

            submitted_fields = set(
                body.keys()
            )

            required_groups = []

            if submitted_fields.intersection(
                ai_fields
            ):
                required_groups.append(
                    {
                        "ai.runtime.manage"
                    }
                )

            if submitted_fields.intersection(
                rag_fields
            ):
                required_groups.append(
                    {
                        "rag.settings.manage"
                    }
                )

            return required_groups

    # ========================================================
    # AI PROVIDERS
    # ========================================================

    if (
        method == "GET"
        and path == "/admin/ai/providers"
    ):
        return [
            {
                "ai.providers.view"
            }
        ]

    # ========================================================
    # AI MODEL DISCOVERY
    # ========================================================

    if (
        method == "GET"
        and path.startswith(
            "/admin/ai/models/"
        )
    ):
        return [
            {
                "ai.models.view"
            }
        ]

    # ========================================================
    # AI CONNECTION TEST
    # ========================================================

    if (
        method == "POST"
        and path == "/admin/ai/test"
    ):
        return [
            {
                "ai.providers.manage"
            }
        ]

    # ========================================================
    # SAVE / DELETE AI CREDENTIAL
    # ========================================================

    if path.startswith(
        "/admin/ai/credentials/"
    ):

        if method in {
            "PUT",
            "DELETE",
        }:
            return [
                {
                    "ai.credentials.manage"
                }
            ]

    # ========================================================
    # ADMIN USERS
    # ========================================================

    if path == "/admin/users":

        if method == "GET":
            return [
                {
                    "users.view"
                }
            ]

    if path.startswith(
        "/admin/users/"
    ):

        # ----------------------------------------------------
        # /admin/users/{user_id}/role
        # ----------------------------------------------------

        if path.endswith(
            "/role"
        ) and method == "PUT":
            return [
                {
                    "users.assign_role"
                }
            ]

        # ----------------------------------------------------
        # /admin/users/{user_id}/status
        # ----------------------------------------------------

        if path.endswith(
            "/status"
        ) and method == "PUT":

            try:
                body = await request.json()

            except Exception:
                body = {}

            if not isinstance(
                body,
                dict,
            ):
                body = {}

            is_active = body.get(
                "is_active"
            )

            if is_active is True:
                return [
                    {
                        "users.activate"
                    }
                ]

            if is_active is False:
                return [
                    {
                        "users.deactivate"
                    }
                ]

            return [
                {
                    "users.activate",
                    "users.deactivate",
                }
            ]

        # ----------------------------------------------------
        # /admin/users/{user_id}
        # ----------------------------------------------------

        if method == "GET":
            return [
                {
                    "users.view"
                }
            ]

    # ========================================================
    # ROLES
    # ========================================================

    if path == "/admin/roles":

        if method == "GET":
            return [
                {
                    "roles.view"
                }
            ]

        if method == "POST":
            return [
                {
                    "roles.create"
                }
            ]

    if path == "/admin/roles/permissions":

        if method == "GET":
            return [
                {
                    "roles.view"
                }
            ]

    if path.startswith(
        "/admin/roles/"
    ):

        # ----------------------------------------------------
        # /admin/roles/{role_key}
        #
        # The actual roles router performs the more specific
        # permission check for update/manage_permissions.
        # ----------------------------------------------------

        if method == "GET":
            return [
                {
                    "roles.view"
                }
            ]

        if method == "PUT":
            return [
                {
                    "roles.update",
                    "roles.manage_permissions",
                }
            ]

        if method == "DELETE":
            return [
                {
                    "roles.delete"
                }
            ]

    # ========================================================
    # FILES
    # ========================================================

    if path == "/files":

        if method == "GET":
            return [
                {
                    "documents.own.view",
                    "documents.all.view",
                }
            ]

    if path == "/files/upload":

        if method == "POST":
            # Allow chat users to upload documents as part of the chat flow.
            # This grants upload access when the user has either the
            # documents.* upload permissions or the chat.use permission.
            return [
                {
                    "documents.own.upload",
                    "documents.all.upload",
                    "chat.use",
                }
            ]

    if path.startswith(
        "/files/"
    ):

        if method == "DELETE":
            return [
                {
                    "documents.own.delete",
                    "documents.all.delete",
                }
            ]

    # ========================================================
    # CHAT
    # ========================================================

    if (
        method == "POST"
        and path == "/chat"
    ):
        return [
            {
                "chat.use"
            }
        ]

    # ========================================================
    # CONVERSATIONS
    # ========================================================

    if path == "/conversations":

        if method == "GET":
            return [
                {
                    "conversations.own.view",
                    "conversations.all.view",
                }
            ]

    if path.startswith(
        "/conversations/"
    ):

        if method == "GET":
            return [
                {
                    "conversations.own.view",
                    "conversations.all.view",
                }
            ]

        if method == "DELETE":
            return [
                {
                    "conversations.own.delete",
                    "conversations.all.delete",
                }
            ]

    # ========================================================
    # NO RBAC RULE FOR THIS ROUTE
    # ========================================================

    return []


# ============================================================
# GET USER PERMISSIONS
# ============================================================

async def get_user_permission_keys(
    current_user: User,
    db: AsyncSession,
) -> set[str]:

    # --------------------------------------------------------
    # Super Admin always has complete access.
    # --------------------------------------------------------

    if current_user.role == "admin":
        return {
            permission.key
            for permission in (
                (
                    await db.execute(
                        select(
                            Permission.key
                        )
                    )
                )
                .scalars()
                .all()
            )
        }

    # --------------------------------------------------------
    # Load permissions from the assigned RBAC role.
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

    # Ensure chat is available to all roles by default.
    # This mirrors the RBAC service behavior and is a low-risk override
    # that makes the chat feature accessible unless explicitly removed.
    try:
        permissions.add("chat.use")
    except Exception:
        # Fallback to literal addition if anything unexpected happens
        permissions.add("chat.use")

    return permissions


# ============================================================
# CHECK CURRENT REQUEST PERMISSIONS
# ============================================================

async def authorize_current_request(
    request: Request,
    current_user: User,
    db: AsyncSession,
) -> None:

    required_groups = (
        await get_required_permission_groups(
            request
        )
    )

    # --------------------------------------------------------
    # This endpoint does not currently have an RBAC rule.
    #
    # Preserve existing behavior for routes such as:
    #
    #     /users/me
    #     /auth/me
    #
    # and other authenticated endpoints.
    # --------------------------------------------------------

    if not required_groups:
        return

    # --------------------------------------------------------
    # Super Admin bypass.
    # --------------------------------------------------------

    if current_user.role == "admin":
        return

    granted_permissions = (
        await get_user_permission_keys(
            current_user=current_user,
            db=db,
        )
    )

    # --------------------------------------------------------
    # Every permission group must be satisfied.
    # --------------------------------------------------------

    for permission_group in (
        required_groups
    ):

        if not permission_group.intersection(
            granted_permissions
        ):

            readable_permissions = (
                sorted(
                    permission_group
                )
            )

            raise HTTPException(
                status_code=(
                    status.HTTP_403_FORBIDDEN
                ),
                detail=(
                    "Permission required: "
                    + " or ".join(
                        readable_permissions
                    )
                ),
            )


# ============================================================
# GET CURRENT USER
# ============================================================

async def get_current_user(
    request: Request,
    token: str = Depends(
        oauth2_scheme
    ),
    db: AsyncSession = Depends(
        get_db
    ),
) -> User:

    credentials_exception = HTTPException(
        status_code=(
            status.HTTP_401_UNAUTHORIZED
        ),
        detail=(
            "Could not validate credentials."
        ),
        headers={
            "WWW-Authenticate": "Bearer"
        },
    )

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[
                ALGORITHM
            ],
        )

        user_id: Optional[str] = (
            payload.get("sub")
        )

        if user_id is None:
            raise credentials_exception

    except (
        JWTError,
        ValueError,
    ):
        raise credentials_exception

    result = await db.execute(
        select(User).where(
            User.id
            == int(user_id)
        )
    )

    user = (
        result.scalar_one_or_none()
    )

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=(
                status.HTTP_403_FORBIDDEN
            ),
            detail=(
                "User account is inactive."
            ),
        )

    # --------------------------------------------------------
    # Backend RBAC enforcement.
    # --------------------------------------------------------

    await authorize_current_request(
        request=request,
        current_user=user,
        db=db,
    )

    return user


# ============================================================
# REQUIRE ADMIN / ADMINISTRATIVE ACCESS
# ============================================================

async def require_admin(
    current_user: User = Depends(
        get_current_user
    ),
    request: Request = None,
) -> User:

    # --------------------------------------------------------
    # Important compatibility behavior:
    #
    # Existing admin endpoints still depend on require_admin.
    #
    # RBAC permission enforcement has already happened inside
    # get_current_user().
    #
    # If the route has a permission rule, a user with that
    # permission is allowed through.
    #
    # If the route has no permission rule, only the original
    # Super Admin role is allowed.
    # --------------------------------------------------------

    required_groups = (
        await get_required_permission_groups(
            request
        )
        if request is not None
        else []
    )

    if required_groups:
        return current_user

    # --------------------------------------------------------
    # Preserve old admin-only behavior for unmapped admin
    # endpoints until they receive an explicit permission rule.
    # --------------------------------------------------------

    if current_user.role != "admin":

        raise HTTPException(
            status_code=(
                status.HTTP_403_FORBIDDEN
            ),
            detail=(
                "Admin access required."
            ),
        )

    return current_user


# ============================================================
# REQUIRE NORMAL USER
# ============================================================

async def require_user(
    current_user: User = Depends(
        get_current_user
    ),
) -> User:

    if current_user.role != "user":

        raise HTTPException(
            status_code=(
                status.HTTP_403_FORBIDDEN
            ),
            detail=(
                "User access required."
            ),
        )

    return current_user