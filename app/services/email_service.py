import resend

from app.config import settings


# ============================================================
# PASSWORD RESET EMAIL
# ============================================================

async def send_password_reset_email(
    recipient_email: str,
    reset_token: str,
) -> None:

    reset_link = (
        "http://localhost:5173/reset-password"
        f"?token={reset_token}"
    )

    resend.api_key = settings.RESEND_API_KEY

    resend.Emails.send(
        {
            "from": settings.EMAIL_FROM,
            "to": [recipient_email],
            "subject": (
                "Reset your Valethi HR Assistant password"
            ),
            "html": f"""
                <div style="
                    font-family: Arial, sans-serif;
                    max-width: 600px;
                    margin: auto;
                    padding: 30px;
                ">

                    <h2>Password Reset Request</h2>

                    <p>
                        We received a request to reset your
                        Valethi HR Assistant password.
                    </p>

                    <p>
                        Click the button below to create
                        a new password:
                    </p>

                    <p>
                        <a
                            href="{reset_link}"
                            style="
                                display:inline-block;
                                padding:12px 20px;
                                background:#2563eb;
                                color:#ffffff;
                                text-decoration:none;
                                border-radius:6px;
                            "
                        >
                            Reset Password
                        </a>
                    </p>

                    <p>
                        This link will expire in
                        <strong>15 minutes</strong>.
                    </p>

                    <p>
                        If you did not request a password
                        reset, you can safely ignore this email.
                    </p>

                    <hr>

                    <p style="color:#666;">
                        Valethi HR Assistant
                    </p>

                </div>
            """,
        }
    )


# ============================================================
# EMAIL VERIFICATION
# ============================================================

async def send_verification_email(
    recipient_email: str,
    verification_token: str,
) -> None:

    verification_link = (
    "http://127.0.0.1:8000/auth/verify-email"
    f"?token={verification_token}"
)

    resend.api_key = settings.RESEND_API_KEY

    resend.Emails.send(
        {
            "from": settings.EMAIL_FROM,
            "to": [recipient_email],
            "subject": (
                "Verify your Valethi HR Assistant account"
            ),
            "html": f"""
                <div style="
                    font-family: Arial, sans-serif;
                    max-width: 600px;
                    margin: auto;
                    padding: 30px;
                ">

                    <h2>
                        Verify your email address
                    </h2>

                    <p>
                        Welcome to Valethi HR Assistant.
                    </p>

                    <p>
                        Please verify your email address
                        by clicking the button below:
                    </p>

                    <p>
                        <a
                            href="{verification_link}"
                            style="
                                display:inline-block;
                                padding:12px 20px;
                                background:#2563eb;
                                color:#ffffff;
                                text-decoration:none;
                                border-radius:6px;
                            "
                        >
                            Verify Email
                        </a>
                    </p>

                    <p>
                        This verification link will expire
                        in <strong>30 minutes</strong>.
                    </p>

                    <p>
                        If you did not create this account,
                        you can safely ignore this email.
                    </p>

                    <hr>

                    <p style="color:#666;">
                        Valethi HR Assistant
                    </p>

                </div>
            """,
        }
    )