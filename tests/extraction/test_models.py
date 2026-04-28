import pytest
from pydantic import ValidationError

from src.extraction.models import DigitalFeature, ExtractionResult, FeatureStatus


def test_digital_feature_valid():
    f = DigitalFeature(
        id="search-product",
        name="Product Search",
        description="Allows users to search for products by keyword.",
        status=FeatureStatus.LIVE,
        parent_product="catalog-api",
        entry_points=["GET /api/products/search"],
        business_capability_hint="Product Discovery",
        confidence_score=0.85,
    )
    assert f.id == "search-product"
    assert f.confidence_score == 0.85


def test_digital_feature_defaults():
    f = DigitalFeature(
        id="x",
        name="X",
        description="desc",
        parent_product="prod",
        entry_points=[],
    )
    assert f.status == FeatureStatus.TO_REVIEW
    assert f.confidence_score == 0.0
    assert f.business_capability_hint is None


def test_digital_feature_confidence_bounds():
    with pytest.raises(ValidationError):
        DigitalFeature(
            id="x", name="X", description="d", parent_product="p",
            entry_points=[], confidence_score=1.5
        )


def test_extraction_result():
    r = ExtractionResult(
        source="https://github.com/org/repo",
        features=[],
        total_clusters=5,
        skipped_clusters=1,
    )
    assert r.total_clusters == 5
