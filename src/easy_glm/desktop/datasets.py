"""Private, session-scoped sources used to score an accepted fit.

The registry intentionally has no connection to ``Project.data``.  Adding a
book to score must never change the modelling source or make a fit stale.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

DatasetSource = Literal["upload", "file", "memory"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path, infer_schema_length=0)
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    raise ValueError("Choose a CSV or Parquet scoring file.")


@dataclass
class ScoringDataset:
    dataset_id: str
    name: str
    source: DatasetSource
    path: str
    digest: str
    version: int
    format: str
    rows: int | None
    columns: list[dict[str, str]]
    filename: str
    status: str = "available"
    message: str | None = None

    def descriptor(self) -> dict[str, Any]:
        return {
            "id": self.dataset_id,
            "name": self.name,
            "source": self.source,
            "path": self.path if self.source == "file" else None,
            "filename": self.filename,
            "format": self.format,
            "digest": self.digest,
            "version": self.version,
            "rows": self.rows,
            "columns": self.columns,
            "status": self.status,
            "message": self.message,
        }


class ScoringDatasets:
    """Own private frame snapshots and verify externally referenced files."""

    def __init__(self) -> None:
        self._folder = tempfile.TemporaryDirectory(prefix="easyglm_scoring_")
        self._items: dict[str, ScoringDataset] = {}

    def close(self) -> None:
        self._folder.cleanup()

    @staticmethod
    def _make(
        *, name: str, source: DatasetSource, path: Path, version: int = 1
    ) -> ScoringDataset:
        frame = _read(path)
        if not frame.width:
            raise ValueError("The scoring data must contain columns.")
        digest = sha256_file(path)
        return ScoringDataset(
            dataset_id=uuid.uuid4().hex,
            name=name.strip() or path.stem,
            source=source,
            path=str(path.resolve()),
            digest=digest,
            version=version,
            format="parquet" if path.suffix.lower() in {".parquet", ".pq"} else "csv",
            rows=frame.height,
            columns=[
                {"name": key, "dtype": str(value)}
                for key, value in frame.schema.items()
            ],
            filename=path.name,
        )

    def add_file(
        self, name: str, path: str | Path, *, replace_id: str | None = None
    ) -> ScoringDataset:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"Scoring file not found: {source}")
        item = self._make(
            name=name,
            source="file",
            path=source,
            version=self._next_version(replace_id),
        )
        return self._replace(item, replace_id)

    def add_upload(
        self,
        name: str,
        filename: str,
        temporary_file: Path,
        *,
        replace_id: str | None = None,
    ) -> ScoringDataset:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".csv", ".parquet", ".pq"}:
            raise ValueError("Choose a CSV or Parquet scoring file.")
        destination = Path(self._folder.name) / (uuid.uuid4().hex + suffix)
        shutil.move(str(temporary_file), destination)
        try:
            item = self._make(
                name=name,
                source="upload",
                path=destination,
                version=self._next_version(replace_id),
            )
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return self._replace(item, replace_id)

    def add_frame(
        self, name: str, frame: pl.DataFrame, *, replace_id: str | None = None
    ) -> ScoringDataset:
        path = Path(self._folder.name) / (uuid.uuid4().hex + ".parquet")
        frame.write_parquet(path)
        item = self._make(
            name=name,
            source="memory",
            path=path,
            version=self._next_version(replace_id),
        )
        return self._replace(item, replace_id)

    def _next_version(self, replace_id: str | None) -> int:
        return (self._items[replace_id].version + 1) if replace_id in self._items else 1

    def _replace(self, item: ScoringDataset, replace_id: str | None) -> ScoringDataset:
        if replace_id:
            old = self._items.pop(replace_id, None)
            if old is None:
                raise ValueError("This scoring dataset is no longer available.")
            item.dataset_id = replace_id
            if old.source in {"upload", "memory"}:
                Path(old.path).unlink(missing_ok=True)
        self._items[item.dataset_id] = item
        return item

    def remove(self, dataset_id: str) -> None:
        item = self._items.pop(dataset_id, None)
        if item is None:
            raise ValueError("This scoring dataset is no longer available.")
        if item.source in {"upload", "memory"}:
            Path(item.path).unlink(missing_ok=True)

    def get(self, dataset_id: str) -> tuple[ScoringDataset, pl.DataFrame]:
        item = self._items.get(dataset_id)
        if item is None:
            raise ValueError("Choose a scoring dataset.")
        path = Path(item.path)
        try:
            actual = sha256_file(path)
        except OSError:
            item.status, item.message = (
                "unavailable",
                "The saved scoring file is unavailable.",
            )
            raise ValueError(item.message) from None
        if actual != item.digest:
            item.status, item.message = (
                "changed",
                "The saved scoring file changed after it was selected.",
            )
            raise ValueError(item.message)
        try:
            frame = _read(path)
        except Exception as exc:
            item.status, item.message = (
                "unavailable",
                f"Could not read scoring data: {exc}",
            )
            raise ValueError(item.message) from exc
        item.status, item.message = "available", None
        return item, frame

    def descriptors(self) -> list[dict[str, Any]]:
        for dataset_id in list(self._items):
            try:
                self.get(dataset_id)
            except ValueError:
                pass
        return [item.descriptor() for item in self._items.values()]

    def restore_descriptor(self, descriptor: dict[str, Any]) -> None:
        """Restore an exported descriptor without silently trusting a changed file."""
        dataset_id = descriptor.get("id")
        path = descriptor.get("path")
        if not isinstance(dataset_id, str) or not isinstance(path, str):
            return
        item = ScoringDataset(
            dataset_id=dataset_id,
            name=str(descriptor.get("name") or Path(path).stem),
            source="file",
            path=path,
            digest=str(descriptor.get("digest") or ""),
            version=int(descriptor.get("version") or 1),
            format=str(descriptor.get("format") or Path(path).suffix.lstrip(".")),
            rows=(
                descriptor.get("rows")
                if isinstance(descriptor.get("rows"), int)
                else None
            ),
            columns=list(descriptor.get("columns") or []),
            filename=str(descriptor.get("filename") or Path(path).name),
        )
        self._items[item.dataset_id] = item
