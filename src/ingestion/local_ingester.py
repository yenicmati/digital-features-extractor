from pathlib import Path

from .base import BaseIngester


class LocalIngester(BaseIngester):
    source_type = "local"

    def ingest(self, source: str) -> list[Path]:
        root = Path(source)
        if not root.exists():
            raise ValueError(f"Path does not exist: {source}")
        all_files = [p for p in root.rglob("*") if p.is_file()]
        return self.filter_files(all_files)
