"""Best-effort framed progress transport for isolated fit workers."""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any, BinaryIO

MAX_PROGRESS_LINE_BYTES = 64 * 1024


class ProgressWriter:
    """Write complete JSON-line packets without affecting the fit on failure."""

    def __init__(
        self, stream: BinaryIO, *, maximum_bytes: int = MAX_PROGRESS_LINE_BYTES
    ):
        self._stream = stream
        self._lock = threading.Lock()
        self._maximum_bytes = maximum_bytes
        self._disabled = False
        self._stream_closed = False

    def __call__(self, message: str | dict[str, Any]) -> None:
        packet = message if isinstance(message, dict) else {"message": message}
        try:
            encoded = (
                json.dumps(packet, allow_nan=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            return
        if len(encoded) > self._maximum_bytes:
            return
        with self._lock:
            if self._disabled or self._stream_closed:
                return
            try:
                remaining = memoryview(encoded)
                while remaining:
                    written = self._stream.write(remaining)
                    if not written:
                        raise OSError("progress pipe closed during write")
                    remaining = remaining[written:]
                self._stream.flush()
            except (OSError, ValueError):
                self._disabled = True

    def close(self) -> None:
        with self._lock:
            if self._stream_closed:
                return
            self._disabled = True
            self._stream_closed = True
            try:
                self._stream.close()
            except (OSError, ValueError):
                pass


def reserve_progress_stdout() -> ProgressWriter:
    """Reserve the stdout pipe, then route ordinary Python/native stdout to stderr."""
    sys.stdout.flush()
    progress_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    progress_stream = os.fdopen(progress_fd, "wb", buffering=0)
    return ProgressWriter(progress_stream)


def progress_packets(
    stream: BinaryIO, *, maximum_bytes: int = MAX_PROGRESS_LINE_BYTES
) -> Iterator[dict[str, Any]]:
    """Yield bounded JSON objects, discarding malformed or oversized frames."""
    while True:
        line = stream.readline(maximum_bytes + 1)
        if not line:
            return
        oversized = len(line) > maximum_bytes
        if oversized and not line.endswith(b"\n"):
            while line and not line.endswith(b"\n"):
                line = stream.readline(maximum_bytes + 1)
        if oversized:
            continue
        try:
            packet = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(packet, dict):
            yield packet


def drain_progress(stream: BinaryIO, receive: Callable[[dict[str, Any]], None]) -> None:
    """Drain a worker pipe continuously, retaining no packet history."""
    try:
        for packet in progress_packets(stream):
            receive(packet)
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except (OSError, ValueError):
            pass
