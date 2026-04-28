from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli import cli
from src.extraction.models import ExtractionResult


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_help_exits_zero(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0


def test_analyze_help_exits_zero(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["analyze", "--help"])
    assert result.exit_code == 0


def test_analyze_nonexistent_source_exits_nonzero(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["analyze", "--source", "./nonexistent_path_xyz"])
    assert result.exit_code != 0
    assert result.output


def _make_extraction_result() -> ExtractionResult:
    return ExtractionResult(
        source="test",
        features=[],
        total_clusters=2,
        skipped_clusters=0,
    )


def test_analyze_valid_local_dir(runner: CliRunner, tmp_path: Path) -> None:
    src_dir = tmp_path / "mysrc"
    src_dir.mkdir()
    (src_dir / "foo.py").write_text("x = 1")

    fake_files = [src_dir / "foo.py"]
    fake_graph = MagicMock()
    fake_graph.number_of_nodes.return_value = 1
    fake_clusters = {"0": ["foo"]}
    fake_result = _make_extraction_result()

    out_dir = tmp_path / "out"

    with (
        patch("src.cli.LocalIngester") as MockLocalIngester,
        patch("src.cli.GraphifyWrapper") as MockWrapper,
        patch("src.cli.FeatureExtractor") as MockExtractor,
        patch("src.cli.JsonExporter") as MockJsonExporter,
        patch("src.cli.HtmlReporter") as MockHtmlReporter,
        patch("src.cli.GraphVisualizer") as MockGraphVisualizer,
        patch("src.cli._build_llm_client", return_value=MagicMock()),
    ):
        MockLocalIngester.return_value.ingest.return_value = fake_files
        MockWrapper.return_value.build_graph.return_value = fake_graph
        MockWrapper.return_value.get_clusters.return_value = fake_clusters
        MockExtractor.return_value.extract.return_value = fake_result

        mock_json_exp = MagicMock()
        MockJsonExporter.return_value = mock_json_exp

        mock_html_rep = MagicMock()
        MockHtmlReporter.return_value = mock_html_rep

        mock_graph_vis = MagicMock()
        MockGraphVisualizer.return_value = mock_graph_vis

        result = runner.invoke(
            cli,
            [
                "analyze",
                "--source",
                str(src_dir),
                "--output-dir",
                str(out_dir),
                "--api-key",
                "sk-test",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "✓ Found" in result.output
    mock_json_exp.export.assert_called_once()
    mock_html_rep.export.assert_called_once()
    mock_graph_vis.export.assert_called_once()
