import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion import GithubIngester, LocalIngester


@pytest.fixture()
def temp_repo():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_local_ingester_returns_supported_files(temp_repo):
    (temp_repo / "main.py").write_text("print('hello')")
    (temp_repo / "app.ts").write_text("const x = 1;")
    (temp_repo / "Main.java").write_text("class Main {}")
    (temp_repo / "readme.txt").write_text("ignore me")

    result = LocalIngester().ingest(str(temp_repo))
    suffixes = {f.suffix for f in result}

    assert ".py" in suffixes
    assert ".ts" in suffixes
    assert ".java" in suffixes
    assert ".txt" not in suffixes
    assert len(result) == 3


def test_local_ingester_raises_on_nonexistent_path():
    with pytest.raises(ValueError, match="does not exist"):
        LocalIngester().ingest("/nonexistent/path/that/does/not/exist")


def test_local_ingester_gitignore_filtering(temp_repo):
    (temp_repo / ".gitignore").write_text("*.pyc\n")
    (temp_repo / "main.py").write_text("code")
    (temp_repo / "main.pyc").write_bytes(b"\x00")

    result = LocalIngester().ingest(str(temp_repo))
    suffixes = {f.suffix for f in result}

    assert ".py" in suffixes
    assert ".pyc" not in suffixes


def test_github_ingester_calls_clone_from(temp_repo):
    url = "https://github.com/example/repo.git"

    (temp_repo / "app.py").write_text("x = 1")
    (temp_repo / "lib.ts").write_text("export {}")
    (temp_repo / "notes.txt").write_text("ignored")

    with patch("src.ingestion.github_ingester.git.Repo.clone_from") as mock_clone:
        mock_clone.return_value = MagicMock()

        ingester = GithubIngester()

        with patch("src.ingestion.github_ingester.tempfile.mkdtemp", return_value=str(temp_repo)):
            result = ingester.ingest(url)

        mock_clone.assert_called_once_with(url, str(temp_repo))

    suffixes = {f.suffix for f in result}
    assert ".py" in suffixes
    assert ".ts" in suffixes
    assert ".txt" not in suffixes


def test_github_ingester_stores_temp_dir():
    url = "https://github.com/example/repo.git"

    with patch("src.ingestion.github_ingester.git.Repo.clone_from"):
        with patch("src.ingestion.github_ingester.tempfile.mkdtemp", return_value="/tmp/fake"):
            ingester = GithubIngester()
            ingester.ingest(url)

    assert ingester.temp_dir == "/tmp/fake"


def test_github_ingester_source_type():
    assert GithubIngester().source_type == "github"


def test_local_ingester_source_type():
    assert LocalIngester().source_type == "local"
