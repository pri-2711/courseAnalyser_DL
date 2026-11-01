"""Password hashing and signed local browser sessions."""
import base64
import hashlib
import hmac
import os
import secrets
import time

import bcrypt


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("ascii")


def verify_password(password: str, stored: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), stored.encode("ascii"))
    except (ValueError, TypeError):
        return False


def _key() -> bytes:
    secret = os.environ.get("COURSE_ANALYZER_SECRET")
    if not secret:
        raise RuntimeError("Set COURSE_ANALYZER_SECRET to a random secret before starting the API.")
    return secret.encode("utf-8")


def sign_session(session_id: str, expires_at: int) -> str:
    payload = f"{session_id}.{expires_at}"
    signature = hmac.new(_key(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}.{signature}".encode()).decode().rstrip("=")


def verify_session(cookie: str):
    try:
        raw = base64.urlsafe_b64decode(cookie + "=" * (-len(cookie) % 4))
        payload, signature = raw.decode().rsplit(".", 1)
        expected = hmac.new(_key(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        session_id, expiry = payload.rsplit(".", 1)
        if int(expiry) <= int(time.time()):
            return None
        return session_id
    except (ValueError, UnicodeError, RuntimeError):
        return None


def new_session():
    session_id = secrets.token_urlsafe(32)
    expires_at = int(time.time()) + 60 * 60 * 24 * 7
    return session_id, expires_at, sign_session(session_id, expires_at)
