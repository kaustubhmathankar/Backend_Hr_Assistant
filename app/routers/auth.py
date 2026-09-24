import secrets
from fastapi.responses import HTMLResponse
from datetime import (
    datetime,
    timedelta,
    timezone,
)

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from fastapi.security import (
    OAuth2PasswordRequestForm,
)

from sqlalchemy import select

from sqlalchemy.ext.asyncio import (
    AsyncSession,
)

from app.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)

from app.database import get_db

from app.rbac_models import Role

from app.services.rbac import (
    get_user_permission_keys,
)

from app.services.email_service import (
    send_password_reset_email,
    send_verification_email,
)

from app.models import (
    EmailVerificationToken,
    PasswordResetToken,
    User,
)

from app.schemas import (
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    ResetPasswordRequest,
    ResetPasswordResponse,
    UserRegister,
    UserResponse,
    VerifyEmailResponse,
)


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


# ============================================================
# REGISTER
# ============================================================

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_user(
    user_data: UserRegister,
    db: AsyncSession = Depends(get_db),
):

    result = await db.execute(
        select(User).where(
            User.email == user_data.email
        )
    )

    existing_user = (
        result.scalar_one_or_none()
    )

    if existing_user:

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "A user with this email "
                "already exists."
            ),
        )

    # --------------------------------------------------------
    # Create new user
    # --------------------------------------------------------

    new_user = User(
        email=user_data.email,
        password_hash=hash_password(
            user_data.password
        ),
        role="user",
        is_active=True,
        email_verified=False,
    )

    db.add(new_user)

    await db.flush()

    # --------------------------------------------------------
    # Generate verification token
    # --------------------------------------------------------

    verification_token = (
        secrets.token_urlsafe(32)
    )

    token_record = (
        EmailVerificationToken(
            token=verification_token,
            user_id=new_user.id,
            expires_at=(
                datetime.now(timezone.utc)
                + timedelta(minutes=30)
            ),
            used=False,
        )
    )

    db.add(token_record)

    await db.commit()

    await db.refresh(new_user)

    # --------------------------------------------------------
    # Send verification email
    # --------------------------------------------------------

    try:

        await send_verification_email(
            recipient_email=new_user.email,
            verification_token=verification_token,
        )

    except Exception as exc:

        # The user and verification token have already
        # been committed. We intentionally don't expose
        # the Resend/API error to the client.

        print(
            "[EMAIL] Verification email failed:",
            exc,
        )

    # --------------------------------------------------------
    # Permissions
    # --------------------------------------------------------

    permissions = await get_user_permission_keys(
        current_user=new_user,
        db=db,
    )

    return {
        "id": new_user.id,
        "email": new_user.email,
        "role": new_user.role,
        "role_name": "User",
        "is_active": new_user.is_active,
        "email_verified": new_user.email_verified,
        "created_at": new_user.created_at,
        "permissions": sorted(permissions),
    }


# ============================================================
# LOGIN
# ============================================================

@router.post(
    "/login"
)
async def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):

    result = await db.execute(
        select(User).where(
            User.email
            == form_data.username
        )
    )

    user = (
        result.scalar_one_or_none()
    )

    if not user:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Invalid email or password."
            ),
        )

    if not verify_password(
        form_data.password,
        user.password_hash,
    ):

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Invalid email or password."
            ),
        )

    if not user.is_active:

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "User account is inactive."
            ),
        )

    # --------------------------------------------------------
    # EMAIL VERIFICATION
    # --------------------------------------------------------

    if not user.email_verified:

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Please verify your email address "
                "before signing in."
            ),
        )

    access_token = (
        create_access_token(
            user_id=user.id,
            email=user.email,
            role=user.role,
        )
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


# ============================================================
# CURRENT USER
# ============================================================

@router.get(
    "/me",
    response_model=UserResponse,
)
async def get_me(
    current_user: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):
    """
    Return the authenticated user's
    profile information.

    Used by the frontend after login
    to determine the logged-in user,
    role, permissions and verification state.
    """

    permissions = await get_user_permission_keys(
        current_user=current_user,
        db=db,
    )

    role_result = await db.execute(
        select(Role.name).where(
            Role.key == current_user.role
        )
    )

    role_name = (
        role_result.scalar_one_or_none()
    )

    if not role_name:

        role_name = (
            "Administrator"
            if current_user.role == "admin"
            else "User"
        )

    return {
        "id": current_user.id,
        "email": current_user.email,
        "role": current_user.role,
        "role_name": role_name,
        "is_active": current_user.is_active,
        "email_verified": current_user.email_verified,
        "created_at": current_user.created_at,
        "permissions": sorted(permissions),
    }


# ============================================================
# VERIFY EMAIL
# ============================================================

@router.get(
    "/verify-email",
    response_class=HTMLResponse,
)
async def verify_email(
    token: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EmailVerificationToken).where(
            EmailVerificationToken.token == token
        )
    )

    verification_token = result.scalar_one_or_none()

    # ------------------------------------------------------------
    # INVALID TOKEN
    # ------------------------------------------------------------
    if not verification_token:
        return HTMLResponse(
            content="""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Verification Failed | Valethi HR Assistant</title>

                <style>
                    * {
                        box-sizing: border-box;
                    }

                    body {
                        margin: 0;
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        padding: 24px;

                        font-family:
                            Inter,
                            -apple-system,
                            BlinkMacSystemFont,
                            "Segoe UI",
                            sans-serif;

                        background:
                            linear-gradient(
                                135deg,
                                #f8fafc 0%,
                                #eef2ff 100%
                            );

                        color: #111827;
                    }

                    .card {
                        width: 100%;
                        max-width: 500px;
                        background: #ffffff;
                        border-radius: 24px;
                        padding: 44px 36px;
                        text-align: center;

                        box-shadow:
                            0 20px 50px rgba(15, 23, 42, 0.10),
                            0 4px 12px rgba(15, 23, 42, 0.05);
                    }

                    .icon {
                        width: 72px;
                        height: 72px;
                        margin: 0 auto 24px;

                        display: flex;
                        align-items: center;
                        justify-content: center;

                        border-radius: 50%;

                        background: #fef2f2;
                        color: #dc2626;

                        font-size: 34px;
                        font-weight: 700;
                    }

                    h1 {
                        margin: 0 0 12px;
                        font-size: 28px;
                        font-weight: 700;
                    }

                    p {
                        margin: 0 auto 28px;
                        max-width: 400px;

                        color: #64748b;
                        font-size: 15px;
                        line-height: 1.7;
                    }

                    .button {
                        display: inline-block;

                        padding: 13px 26px;

                        background: #2563eb;
                        color: #ffffff;

                        text-decoration: none;
                        border-radius: 10px;

                        font-size: 15px;
                        font-weight: 600;

                        transition: background 0.2s ease;
                    }

                    .button:hover {
                        background: #1d4ed8;
                    }

                    .brand {
                        margin-top: 28px;
                        color: #94a3b8;
                        font-size: 13px;
                    }
                </style>
            </head>

            <body>
                <div class="card">

                    <div class="icon">!</div>

                    <h1>Verification Failed</h1>

                    <p>
                        This email verification link is invalid.
                        Please request a new verification email and try again.
                    </p>

                    <a
                        class="button"
                        href="http://localhost:5173/login"
                    >
                        Go to Login
                    </a>

                    <div class="brand">
                        Valethi HR Assistant
                    </div>

                </div>
            </body>
            </html>
            """,
            status_code=400,
        )

    # ------------------------------------------------------------
    # ALREADY USED
    # ------------------------------------------------------------
    if verification_token.used:
        return HTMLResponse(
            content="""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Email Already Verified | Valethi HR Assistant</title>

                <style>
                    * {
                        box-sizing: border-box;
                    }

                    body {
                        margin: 0;
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        padding: 24px;

                        font-family:
                            Inter,
                            -apple-system,
                            BlinkMacSystemFont,
                            "Segoe UI",
                            sans-serif;

                        background: linear-gradient(
                            135deg,
                            #f8fafc,
                            #eef2ff
                        );

                        color: #111827;
                    }

                    .card {
                        width: 100%;
                        max-width: 500px;
                        background: #ffffff;
                        border-radius: 24px;
                        padding: 44px 36px;
                        text-align: center;

                        box-shadow:
                            0 20px 50px rgba(15, 23, 42, 0.10),
                            0 4px 12px rgba(15, 23, 42, 0.05);
                    }

                    .icon {
                        width: 72px;
                        height: 72px;
                        margin: 0 auto 24px;

                        display: flex;
                        align-items: center;
                        justify-content: center;

                        border-radius: 50%;

                        background: #eff6ff;
                        color: #2563eb;

                        font-size: 32px;
                    }

                    h1 {
                        margin: 0 0 12px;
                        font-size: 28px;
                    }

                    p {
                        margin: 0 auto 28px;
                        max-width: 400px;

                        color: #64748b;
                        font-size: 15px;
                        line-height: 1.7;
                    }

                    .button {
                        display: inline-block;
                        padding: 13px 26px;

                        background: #2563eb;
                        color: #ffffff;

                        text-decoration: none;
                        border-radius: 10px;

                        font-size: 15px;
                        font-weight: 600;
                    }

                    .button:hover {
                        background: #1d4ed8;
                    }

                    .brand {
                        margin-top: 28px;
                        color: #94a3b8;
                        font-size: 13px;
                    }
                </style>
            </head>

            <body>
                <div class="card">

                    <div class="icon">✓</div>

                    <h1>Email Already Verified</h1>

                    <p>
                        This email address has already been verified.
                        You can continue to your Valethi HR Assistant account.
                    </p>

                    <a
                        class="button"
                        href="http://localhost:5173/login"
                    >
                        Go to Login
                    </a>

                    <div class="brand">
                        Valethi HR Assistant
                    </div>

                </div>
            </body>
            </html>
            """,
            status_code=400,
        )

    # ------------------------------------------------------------
    # EXPIRED TOKEN
    # ------------------------------------------------------------
    if verification_token.expires_at < datetime.now(timezone.utc):
        return HTMLResponse(
            content="""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Verification Link Expired | Valethi HR Assistant</title>

                <style>
                    * {
                        box-sizing: border-box;
                    }

                    body {
                        margin: 0;
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        padding: 24px;

                        font-family:
                            Inter,
                            -apple-system,
                            BlinkMacSystemFont,
                            "Segoe UI",
                            sans-serif;

                        background: linear-gradient(
                            135deg,
                            #f8fafc,
                            #eef2ff
                        );

                        color: #111827;
                    }

                    .card {
                        width: 100%;
                        max-width: 500px;
                        background: #ffffff;
                        border-radius: 24px;
                        padding: 44px 36px;
                        text-align: center;

                        box-shadow:
                            0 20px 50px rgba(15, 23, 42, 0.10),
                            0 4px 12px rgba(15, 23, 42, 0.05);
                    }

                    .icon {
                        width: 72px;
                        height: 72px;
                        margin: 0 auto 24px;

                        display: flex;
                        align-items: center;
                        justify-content: center;

                        border-radius: 50%;

                        background: #fff7ed;
                        color: #ea580c;

                        font-size: 32px;
                    }

                    h1 {
                        margin: 0 0 12px;
                        font-size: 28px;
                    }

                    p {
                        margin: 0 auto 28px;
                        max-width: 400px;

                        color: #64748b;
                        font-size: 15px;
                        line-height: 1.7;
                    }

                    .button {
                        display: inline-block;
                        padding: 13px 26px;

                        background: #2563eb;
                        color: #ffffff;

                        text-decoration: none;
                        border-radius: 10px;

                        font-size: 15px;
                        font-weight: 600;
                    }

                    .button:hover {
                        background: #1d4ed8;
                    }

                    .brand {
                        margin-top: 28px;
                        color: #94a3b8;
                        font-size: 13px;
                    }
                </style>
            </head>

            <body>
                <div class="card">

                    <div class="icon">⌛</div>

                    <h1>Verification Link Expired</h1>

                    <p>
                        This email verification link has expired.
                        Please request a new verification email to continue.
                    </p>

                    <a
                        class="button"
                        href="http://localhost:5173/login"
                    >
                        Go to Login
                    </a>

                    <div class="brand">
                        Valethi HR Assistant
                    </div>

                </div>
            </body>
            </html>
            """,
            status_code=400,
        )

    # ------------------------------------------------------------
    # FIND USER
    # ------------------------------------------------------------
    result = await db.execute(
        select(User).where(
            User.id == verification_token.user_id
        )
    )

    user = result.scalar_one_or_none()

    if not user:
        return HTMLResponse(
            content="""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Verification Failed | Valethi HR Assistant</title>

                <style>
                    body {
                        margin: 0;
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        padding: 24px;

                        font-family:
                            Inter,
                            -apple-system,
                            BlinkMacSystemFont,
                            "Segoe UI",
                            sans-serif;

                        background: #f8fafc;
                    }

                    .card {
                        width: 100%;
                        max-width: 500px;
                        padding: 44px 36px;
                        background: #ffffff;
                        border-radius: 24px;
                        text-align: center;

                        box-shadow:
                            0 20px 50px rgba(15, 23, 42, 0.10);
                    }

                    h1 {
                        margin-bottom: 12px;
                    }

                    p {
                        color: #64748b;
                        line-height: 1.7;
                        margin-bottom: 28px;
                    }

                    .button {
                        display: inline-block;
                        padding: 13px 26px;
                        background: #2563eb;
                        color: #ffffff;
                        text-decoration: none;
                        border-radius: 10px;
                        font-weight: 600;
                    }
                </style>
            </head>

            <body>
                <div class="card">

                    <h1>Verification Failed</h1>

                    <p>
                        The user account associated with this verification
                        link could not be found.
                    </p>

                    <a
                        class="button"
                        href="http://localhost:5173/login"
                    >
                        Go to Login
                    </a>

                </div>
            </body>
            </html>
            """,
            status_code=400,
        )

    # ------------------------------------------------------------
    # SUCCESSFULLY VERIFY EMAIL
    # ------------------------------------------------------------
    user.email_verified = True
    verification_token.used = True

    await db.commit()

    # ------------------------------------------------------------
    # SUCCESS PAGE
    # ------------------------------------------------------------
    return HTMLResponse(
        content="""
        <!DOCTYPE html>
        <html lang="en">

        <head>
            <meta charset="UTF-8">
            <meta
                name="viewport"
                content="width=device-width, initial-scale=1.0"
            >

            <title>Email Verified | Valethi HR Assistant</title>

            <style>

                * {
                    box-sizing: border-box;
                }

                body {
                    margin: 0;
                    min-height: 100vh;

                    display: flex;
                    align-items: center;
                    justify-content: center;

                    padding: 24px;

                    font-family:
                        Inter,
                        -apple-system,
                        BlinkMacSystemFont,
                        "Segoe UI",
                        sans-serif;

                    background:
                        linear-gradient(
                            135deg,
                            #f8fafc 0%,
                            #eef2ff 100%
                        );

                    color: #111827;
                }

                .card {
                    width: 100%;
                    max-width: 520px;

                    background: #ffffff;

                    border-radius: 24px;

                    padding: 48px 38px;

                    text-align: center;

                    box-shadow:
                        0 25px 60px rgba(15, 23, 42, 0.12),
                        0 5px 15px rgba(15, 23, 42, 0.05);
                }

                .success-icon {
                    width: 82px;
                    height: 82px;

                    margin: 0 auto 26px;

                    display: flex;
                    align-items: center;
                    justify-content: center;

                    border-radius: 50%;

                    background: #dcfce7;

                    color: #16a34a;

                    font-size: 42px;
                    font-weight: 700;
                }

                .brand {
                    font-size: 14px;
                    font-weight: 600;

                    color: #2563eb;

                    margin-bottom: 18px;
                }

                h1 {
                    margin: 0 0 14px;

                    font-size: 30px;
                    line-height: 1.2;

                    font-weight: 700;

                    color: #111827;
                }

                .message {
                    max-width: 420px;

                    margin: 0 auto 30px;

                    color: #64748b;

                    font-size: 15px;

                    line-height: 1.7;
                }

                .button {
                    display: inline-flex;

                    align-items: center;
                    justify-content: center;

                    min-width: 180px;

                    padding: 14px 26px;

                    border-radius: 11px;

                    background: #2563eb;

                    color: #ffffff;

                    text-decoration: none;

                    font-size: 15px;

                    font-weight: 600;

                    box-shadow:
                        0 8px 20px rgba(37, 99, 235, 0.22);

                    transition:
                        background 0.2s ease,
                        transform 0.2s ease,
                        box-shadow 0.2s ease;
                }

                .button:hover {
                    background: #1d4ed8;

                    transform: translateY(-1px);

                    box-shadow:
                        0 10px 24px rgba(37, 99, 235, 0.28);
                }

                .footer {
                    margin-top: 28px;

                    font-size: 12px;

                    color: #94a3b8;
                }

                @media (max-width: 600px) {

                    .card {
                        padding: 38px 24px;
                        border-radius: 20px;
                    }

                    .success-icon {
                        width: 72px;
                        height: 72px;

                        font-size: 36px;
                    }

                    h1 {
                        font-size: 26px;
                    }

                    .message {
                        font-size: 14px;
                    }

                    .button {
                        width: 100%;
                    }
                }

            </style>
        </head>

        <body>

            <div class="card">

                <div class="brand">
                    VALETHI HR ASSISTANT
                </div>

                <div class="success-icon">
                    ✓
                </div>

                <h1>
                    Email Verified Successfully
                </h1>

                <p class="message">
                    Your email address has been successfully verified.
                    Your Valethi HR Assistant account is now ready to use.
                </p>

                <a
                    class="button"
                    href="http://localhost:5173/login"
                >
                    Go to Login
                </a>

                <div class="footer">
                    You can now sign in using your email and password.
                </div>

            </div>

        </body>

        </html>
        """,
        status_code=200,
    )


# ============================================================
# FORGOT PASSWORD
# ============================================================

@router.post(
    "/forgot-password",
    response_model=ForgotPasswordResponse,
)
async def forgot_password(
    request: ForgotPasswordRequest,
    db: AsyncSession = Depends(
        get_db
    ),
):

    result = await db.execute(
        select(User).where(
            User.email == request.email
        )
    )

    user = (
        result.scalar_one_or_none()
    )

    if not user:

        return {
            "message": (
                "If an account with this "
                "email exists, a password "
                "reset email has been sent."
            )
        }

    # --------------------------------------------------------
    # Invalidate old unused tokens
    # --------------------------------------------------------

    existing_tokens = (
        await db.execute(
            select(
                PasswordResetToken
            ).where(
                PasswordResetToken.user_id
                == user.id,
                PasswordResetToken.used.is_(
                    False
                ),
            )
        )
    )

    for old_token in (
        existing_tokens.scalars().all()
    ):

        old_token.used = True

    # --------------------------------------------------------
    # Generate token
    # --------------------------------------------------------

    reset_token = (
        secrets.token_urlsafe(32)
    )

    token_record = (
        PasswordResetToken(
            token=reset_token,
            user_id=user.id,

            expires_at=(
                datetime.now(
                    timezone.utc
                )
                + timedelta(
                    minutes=15
                )
            ),

            used=False,
        )
    )

    db.add(token_record)

    await db.commit()

    # --------------------------------------------------------
    # Send password reset email
    # --------------------------------------------------------

    try:

        await send_password_reset_email(
            recipient_email=user.email,
            reset_token=reset_token,
        )

    except Exception as exc:

        print(
            "[EMAIL] Password reset email failed:",
            exc,
        )

    return {
        "message": (
            "If an account with this email exists, "
            "a password reset email has been sent."
        )
    }


# ============================================================
# RESET PASSWORD
# ============================================================

@router.post(
    "/reset-password",
    response_model=ResetPasswordResponse,
)
async def reset_password(
    request: ResetPasswordRequest,
    db: AsyncSession = Depends(
        get_db
    ),
):

    result = await db.execute(
        select(
            PasswordResetToken
        ).where(
            PasswordResetToken.token
            == request.token
        )
    )

    reset_token = (
        result.scalar_one_or_none()
    )

    if not reset_token:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid password "
                "reset token."
            ),
        )

    if reset_token.used:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "This password reset token "
                "has already been used."
            ),
        )

    # --------------------------------------------------------
    # Token expiration
    # --------------------------------------------------------

    if (
        reset_token.expires_at
        < datetime.now(
            timezone.utc
        )
    ):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "This password reset token "
                "has expired."
            ),
        )

    # --------------------------------------------------------
    # Load user
    # --------------------------------------------------------

    result = await db.execute(
        select(User).where(
            User.id
            == reset_token.user_id
        )
    )

    user = (
        result.scalar_one_or_none()
    )

    if not user:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "User account no longer "
                "exists."
            ),
        )

    # --------------------------------------------------------
    # Update password
    # --------------------------------------------------------

    user.password_hash = (
        hash_password(
            request.new_password
        )
    )

    # --------------------------------------------------------
    # Consume token
    # --------------------------------------------------------

    reset_token.used = True

    await db.commit()

    return {
        "message": (
            "Password reset successfully."
        )
    }