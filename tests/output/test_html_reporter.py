from pathlib import Path

import pytest

from src.extraction.models import DigitalFeature, ExtractionResult, FeatureStatus
from src.output.html_reporter import HtmlReporter


@pytest.fixture()
def sample_result() -> ExtractionResult:
    return ExtractionResult(
        source="test-source",
        total_clusters=3,
        skipped_clusters=0,
        features=[
            DigitalFeature(
                id="f1",
                name="User Authentication",
                description="Allows users to log in securely.",
                status=FeatureStatus.LIVE,
                parent_product="Platform",
                entry_points=["/login"],
                business_capability_hint="Security",
                confidence_score=0.9,
            ),
            DigitalFeature(
                id="f2",
                name="Export to CSV",
                description="Export data as CSV file.",
                status=FeatureStatus.TO_BE_DEVELOPED,
                parent_product="Platform",
                entry_points=["/export"],
                confidence_score=0.55,
            ),
            DigitalFeature(
                id="f3",
                name="Legacy Widget",
                description="Old widget for compatibility.",
                status=FeatureStatus.DEPRECATED,
                parent_product="Platform",
                entry_points=["/widget"],
                confidence_score=0.2,
            ),
        ],
    )


def test_output_file_created(sample_result: ExtractionResult, tmp_path: Path) -> None:
    reporter = HtmlReporter()
    out = tmp_path / "report.html"
    reporter.export(sample_result, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_feature_names_in_html(sample_result: ExtractionResult, tmp_path: Path) -> None:
    reporter = HtmlReporter()
    out = tmp_path / "report.html"
    reporter.export(sample_result, out)
    content = out.read_text(encoding="utf-8")
    assert "User Authentication" in content
    assert "Export to CSV" in content
    assert "Legacy Widget" in content


def test_confidence_scores_in_html(sample_result: ExtractionResult, tmp_path: Path) -> None:
    reporter = HtmlReporter()
    out = tmp_path / "report.html"
    reporter.export(sample_result, out)
    content = out.read_text(encoding="utf-8")
    assert "90%" in content
    assert "55%" in content
    assert "20%" in content


def test_valid_html_structure(sample_result: ExtractionResult, tmp_path: Path) -> None:
    reporter = HtmlReporter()
    out = tmp_path / "report.html"
    reporter.export(sample_result, out)
    content = out.read_text(encoding="utf-8")
    assert "<html" in content
    assert "<head" in content
    assert "<body" in content


def test_report_shows_business_features(tmp_path):
    from src.extraction.models import BusinessFeature, DigitalFeature, ExtractionResult, GroupingResult
    from src.output.html_reporter import HtmlReporter
    f1 = DigitalFeature(id="f1", name="Find spots", description="User can find nearby spots", parent_product="p", entry_points=[], confidence_score=0.9)
    result = ExtractionResult(source="./repo", features=[f1], total_clusters=2, skipped_clusters=0)
    bf = BusinessFeature(id="bf1", name="Spot Discovery", description="Explore sport spots", digital_features=[f1])
    grouping = GroupingResult(source="./repo", business_features=[bf], ungrouped_feature_ids=[])
    out = tmp_path / "report.html"
    HtmlReporter().export(result, out, source="./repo", grouping=grouping)
    html = out.read_text()
    assert "Spot Discovery" in html
    assert "Explore sport spots" in html
