# scripts/auth_cli.py

import os
import requests

# Prefer explicit settings module values (e.g. when running from the repo root),
# but fall back to env vars for flexibility.
try:
    from settings import TOKEN_URL as DEFAULT_TOKEN_URL, CLIENT_ID as DEFAULT_CLIENT_ID, CLIENT_SECRET as DEFAULT_CLIENT_SECRET
except Exception:
    DEFAULT_TOKEN_URL = None
    DEFAULT_CLIENT_ID = None
    DEFAULT_CLIENT_SECRET = None

TOKEN_URL = os.getenv("TOKEN_URL") or DEFAULT_TOKEN_URL
CLIENT_ID = os.getenv("CLIENT_ID") or DEFAULT_CLIENT_ID
CLIENT_SECRET = os.getenv("CLIENT_SECRET") or DEFAULT_CLIENT_SECRET
AUDIENCE = os.getenv("AUDIENCE")  # or scope, depending on your setup


def get_cli_token():
    if not TOKEN_URL:
        raise RuntimeError("TOKEN_URL is not set (env var or settings).")
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError("CLIENT_ID/CLIENT_SECRET are not set (env vars or settings).")

    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "audience": AUDIENCE,
        },
        timeout=30,
    )

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(
            f"Token request failed (status={resp.status_code}): {resp.text.strip()}"
        ) from e

    return resp.json()["access_token"]
    return resp.json()["access_token"]