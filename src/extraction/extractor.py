"""Feature extractor that orchestrates graph clusters + LLM to produce DigitalFeature objects."""

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import ValidationError

from .models import DigitalFeature, ExtractionResult
from .prompts import (
    SYSTEM_PROMPT,
    build_cluster_prompt,
    build_summary_prompt,
    parse_llm_response,
)

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Orchestrates cluster analysis via LLM to extract DigitalFeature objects.

    Args:
        llm_client: OpenAI-compatible client instance (e.g. openai.OpenAI or anthropic.Anthropic).
        model: Model identifier to use for LLM calls.
        cache_dir: If set, LLM responses are cached to/from disk as JSON files.
    """

    def __init__(
        self,
        llm_client: Any,
        model: str = "gpt-4o",
        cache_dir: Path | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _call_llm(self, messages: list[dict]) -> str:
        """Call the LLM and return the raw response text."""
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> str | None:
        path = self._cache_path(key)
        if path is not None and path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def _save_cache(self, key: str, content: str) -> None:
        path = self._cache_path(key)
        if path is not None:
            path.write_text(content, encoding="utf-8")

    def _get_llm_response(self, key: str, messages: list[dict]) -> str:
        """Return cached response or call LLM and cache the result."""
        cached = self._load_cache(key)
        if cached is not None:
            logger.debug("Cache hit for key=%s", key)
            return cached
        content = self._call_llm(messages)
        self._save_cache(key, content)
        return content

    def extract(
        self,
        clusters: dict[str, list[str]],
        graph: nx.Graph,
        source: str = "unknown",
    ) -> ExtractionResult:
        """Extract DigitalFeature objects from graph clusters using LLM.

        Args:
            clusters: Mapping of cluster_id -> list of node names in that cluster.
            graph: NetworkX graph whose node attributes provide 'type' and 'path'.
            source: Source identifier included in the returned ExtractionResult.

        Returns:
            ExtractionResult with deduplicated features and cluster statistics.
        """
        all_raw_features: list[dict] = []
        skipped_clusters = 0
        total_clusters = len(clusters)

        for cluster_id, node_names in clusters.items():
            nodes = []
            for name in node_names:
                attrs = graph.nodes.get(name, {})
                nodes.append(
                    {
                        "name": name,
                        "type": attrs.get("type", "unknown"),
                        "path": attrs.get("path", ""),
                    }
                )

            user_prompt = build_cluster_prompt(cluster_id, nodes)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            try:
                raw_text = self._get_llm_response(cluster_id, messages)
                feature_dicts = parse_llm_response(raw_text)
            except (ValueError, Exception) as exc:
                logger.warning("Skipping cluster %s due to parse error: %s", cluster_id, exc)
                skipped_clusters += 1
                continue

            for fdict in feature_dicts:
                try:
                    feature = DigitalFeature(
                        id=f"{cluster_id}_{fdict.get('name', 'unknown')[:40]}".replace(" ", "_"),
                        name=fdict.get("name", ""),
                        description=fdict.get("description", ""),
                        parent_product=cluster_id,
                        entry_points=[],
                        business_capability_hint=fdict.get("business_capability_hint"),
                        confidence_score=fdict.get("confidence_score", 0.0),
                    )
                    all_raw_features.append(
                        {
                            **fdict,
                            "_validated_id": feature.id,
                            "_parent_product": cluster_id,
                        }
                    )
                except (ValidationError, Exception) as exc:
                    logger.warning(
                        "Skipping invalid feature in cluster %s: %s", cluster_id, exc
                    )
                    skipped_clusters += 1

        final_features = self._deduplicate(all_raw_features, skipped_clusters)

        return ExtractionResult(
            source=source,
            features=final_features,
            total_clusters=total_clusters,
            skipped_clusters=skipped_clusters,
        )

    def _deduplicate(
        self, raw_features: list[dict], skipped_clusters: int
    ) -> list[DigitalFeature]:
        """Run a single LLM call to deduplicate extracted features.

        Falls back to best-effort conversion if LLM or parsing fails.
        """
        if not raw_features:
            return []

        clean_features = [
            {k: v for k, v in f.items() if not k.startswith("_")}
            for f in raw_features
        ]

        summary_prompt = build_summary_prompt(clean_features)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": summary_prompt},
        ]

        try:
            raw_text = self._get_llm_response("__summary__", messages)
            deduped_dicts = parse_llm_response(raw_text)
        except (ValueError, Exception) as exc:
            logger.warning("Deduplication LLM call failed: %s — using raw features", exc)
            deduped_dicts = clean_features

        features: list[DigitalFeature] = []
        for i, fdict in enumerate(deduped_dicts):
            parent = next(
                (
                    r["_parent_product"]
                    for r in raw_features
                    if r.get("name") == fdict.get("name")
                ),
                "unknown",
            )
            try:
                feature = DigitalFeature(
                    id=f"feature_{i}_{fdict.get('name', 'unknown')[:40]}".replace(" ", "_"),
                    name=fdict.get("name", ""),
                    description=fdict.get("description", ""),
                    parent_product=parent,
                    entry_points=[],
                    business_capability_hint=fdict.get("business_capability_hint"),
                    confidence_score=fdict.get("confidence_score", 0.0),
                )
                features.append(feature)
            except (ValidationError, Exception) as exc:
                logger.warning("Skipping invalid deduplicated feature: %s", exc)

        return features
