"""Minimal Postgres access with pg8000 (pure Python, no native libpq needed)."""
import ssl
from urllib.parse import parse_qs, unquote, urlparse

import pg8000.native


def connect(url: str, timeout: int = 15) -> pg8000.native.Connection:
    u = urlparse(url)
    q = parse_qs(u.query)
    ctx = None
    if q.get("sslmode", ["require"])[0] != "disable":
        ctx = ssl.create_default_context()
    return pg8000.native.Connection(
        user=unquote(u.username or ""), password=unquote(u.password or ""), host=u.hostname,
        port=u.port or 5432, database=(u.path or "/neondb").lstrip("/"), ssl_context=ctx, timeout=timeout)
