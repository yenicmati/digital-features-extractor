import json
import pytest

from src.extraction.prompts import (
    SYSTEM_PROMPT,
    build_cluster_prompt,
    build_summary_prompt,
    parse_llm_response,
)


def test_system_prompt_contains_user_concept():
    assert "user" in SYSTEM_PROMPT.lower()


def test_system_prompt_contains_business_value():
    assert "business value" in SYSTEM_PROMPT.lower()


def test_build_cluster_prompt_includes_node_names():
    nodes = [
        {"name": "OrderHistoryController", "type": "class", "path": "src/orders/controller.py"},
        {"name": "search_by_barcode", "type": "function", "path": "src/search/barcode.py"},
    ]
    prompt = build_cluster_prompt("cluster-42", nodes)

    assert "OrderHistoryController" in prompt
    assert "search_by_barcode" in prompt


def test_build_cluster_prompt_includes_cluster_id():
    nodes = [{"name": "Foo", "type": "function", "path": "foo.py"}]
    prompt = build_cluster_prompt("cluster-99", nodes)
    assert "cluster-99" in prompt


def test_parse_llm_response_clean_json():
    features = [{"name": "Track delivery", "confidence_score": 0.9}]
    raw = json.dumps(features)
    result = parse_llm_response(raw)
    assert result == features


def test_parse_llm_response_strips_json_fences():
    features = [{"name": "View order history", "confidence_score": 0.8}]
    raw = "```json\n" + json.dumps(features) + "\n```"
    result = parse_llm_response(raw)
    assert result == features


def test_parse_llm_response_strips_plain_fences():
    features = [{"name": "Search by barcode", "confidence_score": 0.95}]
    raw = "```\n" + json.dumps(features) + "\n```"
    result = parse_llm_response(raw)
    assert result == features


def test_parse_llm_response_raises_value_error_on_invalid_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_llm_response("this is not json at all")


def test_build_summary_prompt_contains_features():
    features = [{"name": "Track delivery", "description": "User tracks their package"}]
    prompt = build_summary_prompt(features)
    assert "Track delivery" in prompt
