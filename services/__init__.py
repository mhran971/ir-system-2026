"""Shared package initialization for the services layer."""

from __future__ import annotations

import sys


def _configure_stdio() -> None:
    """Force UTF-8 text I/O on Windows-friendly consoles."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # If the stream cannot be reconfigured, leave it untouched.
            pass


_configure_stdio()
