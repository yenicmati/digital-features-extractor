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
            entry_points=[], confidence_score=-0.1
        )


def test_digital_feature_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        DigitalFeature(
            id="x",
            name="X",
            description="d",
            parent_product="p",
            entry_points=[],
            unexpected="value",
        )


def test_extraction_result_negative_cluster_count():
    with pytest.raises(ValidationError):
        ExtractionResult(
            source="https://github.com/org/repo",
            features=[],
            total_clusters=-1,
            skipped_clusters=0,
        )


def test_extraction_result():
    r = ExtractionResult(
        source="https://github.com/org/repo",
        features=[],
        total_clusters=5,
        skipped_clusters=1,
    )
    assert r.total_clusters == 5


def test_business_feature_valid():
    from src.extraction.models import BusinessFeature, DigitalFeature

    df = DigitalFeature(
        id="f1", name="Find spots", description="desc",
        parent_product="p", entry_points=[], confidence_score=0.9
    )
    bf = BusinessFeature(
        id="bf1",
        name="Spot Discovery",
        description="All capabilities related to finding sport spots",
        digital_features=[df],
    )
    assert bf.name == "Spot Discovery"
    assert len(bf.digital_features) == 1


def test_business_feature_extra_fields_rejected():
    from src.extraction.models import BusinessFeature
    import pytest

    with pytest.raises(Exception):
        BusinessFeature(id="x", name="x", description="x", digital_features=[], unknown="bad")


def test_grouping_result_valid():
    from src.extraction.models import GroupingResult

    gr = GroupingResult(source="./repo", business_features=[], ungrouped_feature_ids=[])
    assert gr.source == "./repo"
