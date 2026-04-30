from abc import ABC, abstractmethod
from pathlib import Path

import pathspec


class BaseIngester(ABC):
    supported_extensions: frozenset[str] = frozenset(
        {".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".kt", ".vue"}
    )

    @property
    @abstractmethod
    def source_type(self) -> str: ...

    @abstractmethod
    def ingest(self, source: str) -> list[Path]: ...

    def filter_files(self, files: list[Path]) -> list[Path]:
        by_ext = [f for f in files if f.suffix in self.supported_extensions]
        if not by_ext:
            return []

        specs: dict[Path, pathspec.PathSpec | None] = {}

        def _get_spec(directory: Path) -> pathspec.PathSpec | None:
            if directory in specs:
                return specs[directory]
            gitignore = directory / ".gitignore"
            if gitignore.exists():
                spec = pathspec.PathSpec.from_lines(
                    "gitignore", gitignore.read_text().splitlines()
                )
            else:
                spec = None
            specs[directory] = spec
            return spec

        result: list[Path] = []
        for f in by_ext:
            ignored = False
            for parent in [f.parent, *f.parent.parents]:
                spec = _get_spec(parent)
                if spec is not None:
                    if spec.match_file(str(f.relative_to(parent))):
                        ignored = True
                    break
            if not ignored:
                result.append(f)

        return result
