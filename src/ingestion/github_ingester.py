import tempfile
from pathlib import Path

import git

from .base import BaseIngester


class GithubIngester(BaseIngester):
    source_type = "github"

    def __init__(self) -> None:
        self.temp_dir: str | None = None

    def ingest(self, source: str) -> list[Path]:
        self.temp_dir = tempfile.mkdtemp()
        git.Repo.clone_from(source, self.temp_dir)
        all_files = [p for p in Path(self.temp_dir).rglob("*") if p.is_file()]
        return self.filter_files(all_files)
