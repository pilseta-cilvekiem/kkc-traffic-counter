"""Send crossing events to Supabase as they happen.

The board signs in as a dedicated Supabase Auth user (email + password, see `.env.example`)
with the project's *publishable* key, and inserts into `public.crossings` through the REST
API. Row-level security lets only users flagged as uploaders insert, so the credentials on
the board can add counts and do nothing else — the secret key never leaves the workstation.

Sending never blocks counting. Events go into an in-memory queue and a background thread
posts them in batches; when the network or Supabase is down the queue simply grows and is
retried with backoff, so a dropped uplink costs latency, not counts. Each event carries a
client-generated `event_id`, and inserts ignore duplicates on it, so a batch whose response
was lost and is sent again is not counted twice.

The queue is memory only and bounded (`max_queue`): a restart while the uplink is down loses
what was queued, and an outage long enough to fill it drops the oldest events first.

Standard library only — the board's venv has no `requests`.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

TABLE = "crossings"


def load_env_file(path: str | Path) -> None:
    """Put KEY=VALUE lines from `path` into os.environ, without overriding what is set.

    Enough of the dotenv format for a credentials file: comments, blank lines, optional
    `export`, optional quotes. Missing file is not an error.
    """
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(key, value)


class SupabaseReporter:
    """Queue events and post them to Supabase from a background thread."""

    def __init__(
        self,
        url: str,
        publishable_key: str,
        email: str,
        password: str,
        *,
        batch_size: int = 200,
        max_queue: int = 100_000,
        max_backoff: float = 300.0,
        timeout: float = 15.0,
    ):
        self.url = url.rstrip("/")
        self.key = publishable_key
        self.email = email
        self.password = password
        self.batch_size = batch_size
        self.max_backoff = max_backoff
        self.timeout = timeout

        self._queue: deque[dict] = deque(maxlen=max_queue)
        self._cond = threading.Condition()
        self._stopping = False
        self._token: str | None = None
        self._refresh_token: str | None = None
        self._token_expires = 0.0
        self._failing = False

        self.sent = 0
        self.dropped = 0

        self._thread = threading.Thread(target=self._run, name="supabase-report", daemon=True)
        self._thread.start()

    @classmethod
    def from_env(cls) -> "SupabaseReporter":
        missing = [
            k for k in (
                "SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY",
                "SUPABASE_UPLOADER_EMAIL", "SUPABASE_UPLOADER_PASSWORD",
            )
            if not os.environ.get(k)
        ]
        if missing:
            raise RuntimeError(f"--report needs {', '.join(missing)} (see .env.example)")
        return cls(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_PUBLISHABLE_KEY"],
            os.environ["SUPABASE_UPLOADER_EMAIL"],
            os.environ["SUPABASE_UPLOADER_PASSWORD"],
        )

    @property
    def pending(self) -> int:
        with self._cond:
            return len(self._queue)

    def submit(self, event: dict, occurred_at: float) -> None:
        """Queue one `CrossingEvent.as_dict()`; `occurred_at` is a unix time."""
        # Only what a count needs; the full event stays in the local --events file.
        row = {
            "event_id": str(uuid.uuid4()),
            "occurred_at": datetime.fromtimestamp(occurred_at, timezone.utc).isoformat(),
            "class_name": event["class"],
            "direction": event["direction"],
        }
        with self._cond:
            if len(self._queue) == self._queue.maxlen:
                self.dropped += 1
            self._queue.append(row)
            self._cond.notify()

    def close(self, timeout: float = 10.0) -> None:
        """Try to send what is queued, for at most `timeout` seconds."""
        with self._cond:
            self._stopping = True
            self._cond.notify()
        self._thread.join(timeout)

    # --- background thread ------------------------------------------------------------

    def _run(self) -> None:
        backoff = 1.0
        while True:
            with self._cond:
                while not self._queue and not self._stopping:
                    self._cond.wait()
                if not self._queue:
                    return
                batch = [self._queue[i] for i in range(min(self.batch_size, len(self._queue)))]

            try:
                self._insert(batch)
            except Exception as e:  # noqa: BLE001 — any failure means "keep it and retry"
                if not self._failing:
                    _log(f"send failed, keeping {self.pending} event(s) to retry: {e}")
                    self._failing = True
                if self._stopping:
                    return
                with self._cond:
                    self._cond.wait(backoff)
                backoff = min(backoff * 2, self.max_backoff)
                continue

            with self._cond:
                # Only events from the head were sent, and only this thread pops, so the
                # first len(batch) entries are still those — unless the full deque evicted
                # some of them meanwhile, in which case fewer remain to pop.
                for row in batch:
                    if self._queue and self._queue[0] is row:
                        self._queue.popleft()
            self.sent += len(batch)
            backoff = 1.0
            if self._failing:
                _log(f"sending again, {self.pending} event(s) still queued")
                self._failing = False

    def _insert(self, rows: list[dict]) -> None:
        try:
            self._request(
                f"/rest/v1/{TABLE}?on_conflict=event_id",
                rows,
                {"Prefer": "return=minimal,resolution=ignore-duplicates"},
                auth=True,
            )
        except urllib.error.HTTPError as e:
            if e.code == 401:
                # Expired or revoked; sign in from scratch on the next attempt.
                self._token = self._refresh_token = None
            raise

    def _ensure_token(self) -> str:
        if self._token and time.time() < self._token_expires - 60:
            return self._token
        body = None
        if self._refresh_token:
            try:
                body = self._request(
                    "/auth/v1/token?grant_type=refresh_token",
                    {"refresh_token": self._refresh_token},
                )
            except urllib.error.HTTPError:
                body = None  # refresh token used up or revoked; fall back to the password
        if body is None:
            body = self._request(
                "/auth/v1/token?grant_type=password",
                {"email": self.email, "password": self.password},
            )
        self._token = body["access_token"]
        self._refresh_token = body.get("refresh_token")
        self._token_expires = time.time() + float(body.get("expires_in", 3600))
        return self._token

    def _request(self, path: str, payload, headers: dict | None = None, auth: bool = False):
        h = {"apikey": self.key, "Content-Type": "application/json"}
        if auth:
            h["Authorization"] = f"Bearer {self._ensure_token()}"
        h.update(headers or {})
        req = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode(), headers=h, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            # The response body says *why* (RLS violation, bad column, wrong password).
            detail = e.read().decode(errors="replace")[:300]
            e.msg = f"{e.msg}: {detail}"
            raise
        return json.loads(data) if data else None


def _log(msg: str) -> None:
    print(f"\n[report] {msg}", file=sys.stderr, flush=True)
