"""Post Discord webhook bodies, with the standard library only.

The webhook URL is a secret: it is read from an environment variable, never printed, and scrubbed from any
message that leaves this module. An HTTP 429 is retried after the `retry_after` Discord names
(https://discord.com/developers/docs/topics/rate-limits); any other non-2xx fails with the response body.
"""

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

MAX_RETRIES = 5
MAX_WAIT_S = 60.0
PAUSE_S = 0.5  # between messages, well inside Discord's 5 requests per 2 seconds per webhook

# (url, JSON body) -> (status, response body); a transport failure raises OSError.
Send = Callable[[str, bytes], tuple[int, str]]


class NotifyError(RuntimeError):
    """A post failed; the message never holds the webhook URL."""


def urllib_send(url: str, body: bytes) -> tuple[int, str]:
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json", "User-Agent": "graph-bench"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def _retry_after(body: str) -> float:
    try:
        value = float(json.loads(body).get("retry_after", 1.0))
    except (ValueError, AttributeError, TypeError):
        value = 1.0
    return min(max(value, 0.0), MAX_WAIT_S)


def post_all(
    url: str,
    payloads: list[dict[str, Any]],
    *,
    send: Send | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """POST each payload in order; return how many were sent. Raises NotifyError on a failure."""

    send = send or urllib_send  # looked up at call time, so tests can replace it

    def scrub(text: str) -> str:
        return text.replace(url, "<webhook>")

    for index, payload in enumerate(payloads, start=1):
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(MAX_RETRIES + 1):
            try:
                status, text = send(url, body)
            except OSError as exc:
                raise NotifyError(scrub(f"message {index}: request failed: {type(exc).__name__}")) from None
            if 200 <= status < 300:
                break
            if status == 429 and attempt < MAX_RETRIES:
                sleep(_retry_after(text))
                continue
            raise NotifyError(scrub(f"message {index}: HTTP {status}: {text.strip()[:500]}"))
        if index < len(payloads):
            sleep(PAUSE_S)
    return len(payloads)
