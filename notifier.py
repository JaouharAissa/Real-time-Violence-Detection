import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional


def _env(key: str, default: str = "") -> str:
    v = os.environ.get(key)
    return v if isinstance(v, str) and v.strip() else default


def _bool_env(key: str, default: bool = False) -> bool:
    v = _env(key, "")
    if not v:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def send_violence_email(
    *,
    subject: str,
    body: str,
    to_email: Optional[str] = None,
) -> bool:
    """
    Send a best-effort email notification.

    Controlled by env:
      - NOTIFY_EMAIL_ENABLED=true|false
      - NOTIFY_EMAIL_TO=owner@example.com   (if to_email not provided)
      - SMTP_HOST, SMTP_PORT
      - SMTP_USER, SMTP_PASSWORD
      - SMTP_FROM
      - SMTP_STARTTLS=true|false
    """
    if not _bool_env("NOTIFY_EMAIL_ENABLED", False):
        return False

    host = _env("SMTP_HOST")
    port = int(_env("SMTP_PORT", "587"))
    user = _env("SMTP_USER")
    password = _env("SMTP_PASSWORD")
    sender = _env("SMTP_FROM", user)
    recipient = (to_email or _env("NOTIFY_EMAIL_TO")).strip()
    starttls = _bool_env("SMTP_STARTTLS", True)

    if not (host and sender and recipient):
        return False

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    context = ssl.create_default_context()

    try:
        with smtplib.SMTP(host=host, port=port, timeout=10) as smtp:
            smtp.ehlo()
            if starttls:
                smtp.starttls(context=context)
                smtp.ehlo()
            if user and password:
                smtp.login(user, password)
            smtp.send_message(msg)
        return True
    except Exception:
        return False

