import json
from pathlib import Path

import pytest

from src.extraction.models import DigitalFeature, ExtractionResult, FeatureStatus
from src.output.json_exporter import JsonExporter


def make_result() -> ExtractionResult:
    return ExtractionResult(
        source="test_source",
        features=[
            DigitalFeature(
                id="f1",
                name="Feature One",
                description="Desc one",
                status=FeatureStatus.LIVE,
                parent_product="Product A",
                entry_points=["/a"],
                confidence_score=0.6,
            ),
            DigitalFeature(
                id="f2",
                name="Feature Two",
                description="Desc two",
                status=FeatureStatus.TO_BE_DEVELOPED,
                parent_product="Product B",
                entry_points=["/b"],
                confidence_score=0.9,
            ),
            DigitalFeature(
                id="f3",
                name="Feature Three",
                description="Desc three",
                status=FeatureStatus.DEPRECATED,
                parent_product="Product A",
                entry_points=["/c"],
                confidence_score=0.3,
            ),
        ],
        total_clusters=10,
        skipped_clusters=1,
    )


def test_output_file_created(tmp_path: Path) -> None:
    exporter = JsonExporter()
    out = tmp_path / "features.json"
    exporter.export(make_result(), out)
    assert out.exists()


def test_output_is_valid_json(tmp_path: Path) -> None:
    exporter = JsonExporter()
    out = tmp_path / "features.json"
    exporter.export(make_result(), out)
    data = json.loads(out.read_text())
    assert isinstance(data, dict)


def test_metadata_fields_present(tmp_path: Path) -> None:
    exporter = JsonExporter()
    out = tmp_path / "features.json"
    exporter.export(make_result(), out)
    data = json.loads(out.read_text())
    meta = data["metadata"]
    assert "exported_at" in meta
    assert meta["total_features"] == 3
    assert meta["total_clusters"] == 10
    assert meta["skipped_clusters"] == 1


def test_features_sorted_by_confidence_descending(tmp_path: Path) -> None:
    exporter = JsonExporter()
    out = tmp_path / "features.json"
    exporter.export(make_result(), out)
    data = json.loads(out.read_text())
    scores = [f["confidence_score"] for f in data["features"]]
    assert scores == sorted(scores, reverse=True)


def test_correct_structure(tmp_path: Path) -> None:
    exporter = JsonExporter()
    out = tmp_path / "features.json"
    exporter.export(make_result(), out)
    data = json.loads(out.read_text())
    assert "metadata" in data
    assert "features" in data
    assert len(data["features"]) == 3
    first = data["features"][0]
    assert first["id"] == "f2"


def test_export_with_grouping_result(tmp_path):
    import json
    from src.extraction.models import BusinessFeature, DigitalFeature, ExtractionResult, GroupingResult
    from src.output.json_exporter import JsonExporter
    f1 = DigitalFeature(id="f1", name="Find spots", description="desc", parent_product="p", entry_points=[], confidence_score=0.9)
    result = ExtractionResult(source="./repo", features=[f1], total_clusters=2, skipped_clusters=0)
    bf = BusinessFeature(id="bf1", name="Spot Discovery", description="desc", digital_features=[f1])
    grouping = GroupingResult(source="./repo", business_features=[bf], ungrouped_feature_ids=[])
    out = tmp_path / "features.json"
    JsonExporter().export(result, out, grouping=grouping)
    data = json.loads(out.read_text())
    assert "business_features" in data
    assert data["business_features"][0]["name"] == "Spot Discovery"
    assert len(data["business_features"][0]["digital_features"]) == 1
