import json

import pytest

from src.extraction.grouping_prompts import build_grouping_prompt, parse_grouping_response


SAMPLE_FEATURES = [
    {"id": "f1", "name": "Find spots near me", "business_capability_hint": "Spot Discovery"},
    {"id": "f2", "name": "View spot details", "business_capability_hint": "Spot Discovery"},
    {"id": "f3", "name": "Submit condition report", "business_capability_hint": "Community Reporting"},
]


def test_grouping_prompt_contains_feature_names():
    prompt = build_grouping_prompt(SAMPLE_FEATURES)
    assert "Find spots near me" in prompt
    assert "Submit condition report" in prompt


def test_grouping_prompt_contains_ids():
    prompt = build_grouping_prompt(SAMPLE_FEATURES)
    assert "f1" in prompt
    assert "f3" in prompt


def test_parse_grouping_response_valid():
    raw = json.dumps([{"name": "Spot Discovery", "description": "Finding spots", "feature_ids": ["f1", "f2"]}])
    groups = parse_grouping_response(raw)
    assert len(groups) == 1
    assert groups[0]["name"] == "Spot Discovery"
    assert "f1" in groups[0]["feature_ids"]


def test_parse_grouping_response_strips_fences():
    raw = '```json\n[{"name":"X","description":"Y","feature_ids":["f1"]}]\n```'
    groups = parse_grouping_response(raw)
    assert groups[0]["name"] == "X"


def test_parse_grouping_response_invalid_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_grouping_response("not json at all")
