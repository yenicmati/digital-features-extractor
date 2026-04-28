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
