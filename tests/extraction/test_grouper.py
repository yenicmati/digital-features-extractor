import json
import pytest
from unittest.mock import MagicMock
from src.extraction.grouper import FeatureGrouper
from src.extraction.models import DigitalFeature, ExtractionResult


def _make_feature(id: str, name: str, hint: str | None = None) -> DigitalFeature:
    return DigitalFeature(id=id, name=name, description="desc", parent_product="p", entry_points=[], confidence_score=0.8, business_capability_hint=hint)


def _make_result(features):
    return ExtractionResult(source="./repo", features=features, total_clusters=5, skipped_clusters=0)


def _mock_llm(response_text: str):
    client = MagicMock()
    msg = MagicMock(); msg.content = response_text
    choice = MagicMock(); choice.message = msg
    completion = MagicMock(); completion.choices = [choice]
    client.chat.completions.create.return_value = completion
    return client


def test_group_returns_grouping_result():
    features = [_make_feature("f1","Find spots","Spot"), _make_feature("f2","View spot","Spot"), _make_feature("f3","Submit report","Community"), _make_feature("f4","View reports","Community")]
    llm_resp = json.dumps([{"name":"Spot","description":"Find","feature_ids":["f1","f2"]},{"name":"Community","description":"Reports","feature_ids":["f3","f4"]}])
    grouping = FeatureGrouper(_mock_llm(llm_resp), "gpt-4.1").group(_make_result(features))
    assert len(grouping.business_features) == 2
    assert grouping.source == "./repo"


def test_group_ungrouped_feature_ids():
    features = [_make_feature("f1","Find spots"), _make_feature("f2","Other")]
    llm_resp = json.dumps([{"name":"Spot","description":"desc","feature_ids":["f1"]}])
    grouping = FeatureGrouper(_mock_llm(llm_resp), "gpt-4.1").group(_make_result(features))
    assert "f2" in grouping.ungrouped_feature_ids


def test_group_empty_features():
    grouping = FeatureGrouper(MagicMock(), "gpt-4.1").group(_make_result([]))
    assert grouping.business_features == []


def test_group_llm_failure_uses_fallback():
    features = [_make_feature("f1","Find","Spot"), _make_feature("f2","View","Spot"), _make_feature("f3","Report","Community")]
    client = MagicMock(); client.chat.completions.create.side_effect = Exception("down")
    grouping = FeatureGrouper(client, "gpt-4.1").group(_make_result(features))
    assert len(grouping.business_features) >= 1
