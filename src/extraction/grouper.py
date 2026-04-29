from __future__ import annotations
import logging
from typing import Any
from .grouping_prompts import GROUPING_SYSTEM_PROMPT, build_grouping_prompt, parse_grouping_response
from .models import BusinessFeature, DigitalFeature, ExtractionResult, GroupingResult

logger = logging.getLogger(__name__)

class FeatureGrouper:
    def __init__(self, llm_client: Any, model: str = "gpt-4.1") -> None:
        self.llm_client = llm_client
        self.model = model

    def group(self, result: ExtractionResult) -> GroupingResult:
        if not result.features:
            return GroupingResult(source=result.source, business_features=[], ungrouped_feature_ids=[])

        feature_dicts = [
            {"id": f.id, "name": f.name, "description": f.description, "business_capability_hint": f.business_capability_hint}
            for f in result.features
        ]

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": GROUPING_SYSTEM_PROMPT},
                    {"role": "user", "content": build_grouping_prompt(feature_dicts)},
                ],
            )
            raw_text = response.choices[0].message.content
            groups = parse_grouping_response(raw_text)
        except Exception as exc:
            logger.warning("LLM grouping failed: %s — falling back to hint-based grouping", exc)
            return self._fallback_group(result)

        feature_by_id = {f.id: f for f in result.features}
        assigned_ids: set[str] = set()
        business_features: list[BusinessFeature] = []

        for i, group in enumerate(groups):
            ids = group.get("feature_ids", [])
            matched = [feature_by_id[fid] for fid in ids if fid in feature_by_id]
            if not matched:
                continue
            assigned_ids.update(fid for fid in ids if fid in feature_by_id)
            business_features.append(BusinessFeature(
                id=f"bf_{i}_{group.get('name', 'group')[:30]}".replace(" ", "_"),
                name=group.get("name", f"Group {i}"),
                description=group.get("description", ""),
                digital_features=matched,
            ))

        ungrouped = [f.id for f in result.features if f.id not in assigned_ids]
        return GroupingResult(source=result.source, business_features=business_features, ungrouped_feature_ids=ungrouped)

    def _fallback_group(self, result: ExtractionResult) -> GroupingResult:
        from collections import defaultdict
        buckets: dict[str, list[DigitalFeature]] = defaultdict(list)
        for f in result.features:
            key = f.business_capability_hint or "Other"
            buckets[key].append(f)
        business_features = [
            BusinessFeature(
                id=f"bf_{i}_{name[:30]}".replace(" ", "_"),
                name=name,
                description=f"Features related to {name}",
                digital_features=features,
            )
            for i, (name, features) in enumerate(buckets.items())
        ]
        return GroupingResult(source=result.source, business_features=business_features, ungrouped_feature_ids=[])
