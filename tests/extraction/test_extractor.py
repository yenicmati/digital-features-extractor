import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest
from pydantic import ValidationError

from src.extraction.extractor import FeatureExtractor
from src.extraction.models import DigitalFeature, ExtractionResult


VALID_FEATURE = {
    "name": "View order history",
    "description": "User can see past orders",
    "user_value": "Helps track purchases",
    "confidence_score": 0.9,
    "business_capability_hint": "Order Management",
}

VALID_RESPONSE = json.dumps([VALID_FEATURE])


def make_llm_client(response_text: str) -> MagicMock:
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


def make_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_node("OrderService", type="class", path="src/orders/service.py")
    g.add_node("OrderController", type="class", path="src/orders/controller.py")
    return g


def make_clusters() -> dict[str, list[str]]:
    return {"cluster_0": ["OrderService", "OrderController"]}


def test_valid_response_returns_features():
    client = make_llm_client(VALID_RESPONSE)
    extractor = FeatureExtractor(llm_client=client, model="gpt-4o")
    result = extractor.extract(make_clusters(), make_graph(), source="test")

    assert isinstance(result, ExtractionResult)
    assert result.total_clusters == 1
    assert len(result.features) >= 1
    assert all(isinstance(f, DigitalFeature) for f in result.features)
    assert result.features[0].name == "View order history"


def test_invalid_json_increments_skipped_clusters():
    client = make_llm_client("not valid json at all")
    extractor = FeatureExtractor(llm_client=client, model="gpt-4o")
    result = extractor.extract(make_clusters(), make_graph(), source="test")

    assert result.total_clusters == 1
    assert result.skipped_clusters == 1
    assert result.features == []


def test_cache_miss_calls_llm(tmp_path: Path):
    client = make_llm_client(VALID_RESPONSE)
    extractor = FeatureExtractor(llm_client=client, model="gpt-4o", cache_dir=tmp_path)
    extractor.extract(make_clusters(), make_graph(), source="test")

    assert client.chat.completions.create.called


def test_cache_hit_skips_llm(tmp_path: Path):
    (tmp_path / "__micro_merged__.json").write_text(VALID_RESPONSE, encoding="utf-8")
    (tmp_path / "__summary__.json").write_text(VALID_RESPONSE, encoding="utf-8")

    client = make_llm_client(VALID_RESPONSE)
    extractor = FeatureExtractor(llm_client=client, model="gpt-4o", cache_dir=tmp_path)
    extractor.extract(make_clusters(), make_graph(), source="test")

    client.chat.completions.create.assert_not_called()


def test_validation_error_on_feature_skips_gracefully():
    bad_feature = {
        "name": "Bad Feature",
        "description": "Has invalid confidence",
        "user_value": "Some value",
        "confidence_score": 2.5,
        "business_capability_hint": None,
    }
    client = make_llm_client(json.dumps([bad_feature]))
    extractor = FeatureExtractor(llm_client=client, model="gpt-4o")
    result = extractor.extract(make_clusters(), make_graph(), source="test")

    assert result.total_clusters == 1
    assert result.skipped_clusters == 1


def test_cache_written_on_miss(tmp_path: Path):
    client = make_llm_client(VALID_RESPONSE)
    extractor = FeatureExtractor(llm_client=client, model="gpt-4o", cache_dir=tmp_path)
    extractor.extract(make_clusters(), make_graph(), source="test")

    assert (tmp_path / "__micro_merged__.json").exists()
    cached = (tmp_path / "__micro_merged__.json").read_text(encoding="utf-8")
    assert cached == VALID_RESPONSE


def test_multiple_clusters_total_count():
    client = make_llm_client(VALID_RESPONSE)
    g = make_graph()
    g.add_node("ProductService", type="class", path="src/products/service.py")
    clusters = {
        "cluster_0": ["OrderService", "OrderController"],
        "cluster_1": ["ProductService"],
    }
    extractor = FeatureExtractor(llm_client=client, model="gpt-4o")
    result = extractor.extract(clusters, g, source="test")

    assert result.total_clusters == 2


def test_micro_clusters_merged_reduces_llm_calls():
    client = make_llm_client(VALID_RESPONSE)
    g = nx.Graph()
    for i in range(5):
        g.add_node(f"Node{i}", type="function", path=f"src/mod{i}.py")
    clusters = {f"cluster_{i}": [f"Node{i}"] for i in range(5)}

    extractor = FeatureExtractor(llm_client=client, model="gpt-4o")
    result = extractor.extract(clusters, g, source="test")

    assert result.total_clusters == 5
    call_count = client.chat.completions.create.call_count
    assert call_count < 5 + 1
