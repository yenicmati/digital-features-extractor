"""Feature extractor that orchestrates graph clusters + LLM to produce DigitalFeature objects."""

import json
import logging
import re
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import ValidationError

from .models import DigitalFeature, ExtractionResult
from .prompts import (
    PREFILTER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_cluster_prompt,
    build_prefilter_prompt,
    build_routes_prompt,
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
        cached = self._load_cache(key)
        if cached is not None:
            logger.debug("Cache hit for key=%s", key)
            return cached
        content = self._call_llm(messages)
        self._save_cache(key, content)
        return content

    _VUE_TEMPLATE_RE = re.compile(
        r"<template[^>]*>(.*?)</template>", re.DOTALL | re.IGNORECASE
    )
    _VUE_SCRIPT_RE = re.compile(
        r"<script[^>]*>(.*?)</script>", re.DOTALL | re.IGNORECASE
    )

    def _extract_file_content(self, file_path: Path) -> str | None:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None

        if file_path.suffix == ".vue":
            parts: list[str] = []
            tm = self._VUE_TEMPLATE_RE.search(text)
            if tm:
                parts.append("[template]\n" + tm.group(1).strip()[:800])
            sm = self._VUE_SCRIPT_RE.search(text)
            if sm:
                parts.append("[script]\n" + "\n".join(sm.group(1).strip().splitlines()[:30]))
            return "\n\n".join(parts) if parts else "\n".join(text.splitlines()[:60])

        return "\n".join(text.splitlines()[:60])

    _ROUTER_FILE_STEMS = frozenset({
        "router", "routes", "index", "app-routing.module",
        "app.router", "routing",
    })
    _ROUTE_PATH_RE = re.compile(r"""path\s*:\s*['"]([^'"]+)['"]""")
    _ROUTE_NAME_RE = re.compile(r"""name\s*:\s*['"]([^'"]+)['"]""")
    _PREFILTER_MIN_CLUSTERS = 6

    def _extract_routes_from_files(self, files: list[Path]) -> list[dict]:
        router_files = [
            f for f in files
            if f.stem.lower().replace("-", "").replace(".", "") in {
                n.replace("-", "").replace(".", "") for n in self._ROUTER_FILE_STEMS
            }
            and f.suffix in {".ts", ".js", ".vue"}
        ]
        routes: list[dict] = []
        for rf in router_files:
            try:
                text = rf.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            paths = self._ROUTE_PATH_RE.findall(text)
            names = self._ROUTE_NAME_RE.findall(text)
            for i, path in enumerate(paths):
                routes.append({"path": path, "name": names[i] if i < len(names) else path.strip("/")})
        return routes

    def _prefilter_clusters(self, summaries: dict[str, str]) -> list[str]:
        prompt = build_prefilter_prompt(summaries)
        messages = [
            {"role": "system", "content": PREFILTER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = self._get_llm_response("__prefilter__", messages)
            parsed = parse_llm_response(raw)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return parsed
            return list(summaries.keys())
        except Exception:
            return list(summaries.keys())

    def extract(
        self,
        clusters: dict[str, list[str]],
        graph: nx.Graph,
        source: str = "unknown",
        project_context: str | None = None,
        files: list[Path] | None = None,
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

        system_content = SYSTEM_PROMPT
        if project_context:
            system_content = f"{SYSTEM_PROMPT}\n\n{project_context}"

        micro_nodes: list[str] = []
        clusters_to_process: dict[str, list[str]] = {}
        for cluster_id, node_names in clusters.items():
            if len(node_names) <= 2:
                micro_nodes.extend(node_names)
            else:
                clusters_to_process[cluster_id] = node_names
        if micro_nodes:
            clusters_to_process["__micro_merged__"] = micro_nodes

        if len(clusters_to_process) >= self._PREFILTER_MIN_CLUSTERS:
            cluster_summaries = {
                cid: ", ".join(node_names[:5])
                for cid, node_names in clusters_to_process.items()
            }
            kept_ids = self._prefilter_clusters(cluster_summaries)
            clusters_to_process = {k: v for k, v in clusters_to_process.items() if k in kept_ids}
            logger.debug("Pre-filter: kept %d / %d clusters", len(clusters_to_process), len(cluster_summaries))

        for cluster_id, node_names in clusters_to_process.items():
            nodes = []
            for name in node_names:
                attrs = graph.nodes.get(name, {})
                path_str = attrs.get("path", "")
                content: str | None = None
                if path_str:
                    fp = Path(path_str)
                    if fp.exists():
                        content = self._extract_file_content(fp)
                nodes.append(
                    {
                        "name": name,
                        "type": attrs.get("type", "unknown"),
                        "path": path_str,
                        "content": content,
                    }
                )

            user_prompt = build_cluster_prompt(cluster_id, nodes)
            messages = [
                {"role": "system", "content": system_content},
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

        final_features = self._deduplicate(all_raw_features, skipped_clusters, system_content)

        routes = self._extract_routes_from_files(files) if files else []
        if routes:
            routes_messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": build_routes_prompt(routes)},
            ]
            try:
                raw_text = self._get_llm_response("__routes__", routes_messages)
                route_dicts = parse_llm_response(raw_text)
                route_features: list[DigitalFeature] = []
                for i, fdict in enumerate(route_dicts):
                    try:
                        route_features.append(DigitalFeature(
                            id=f"route_{i}_{fdict.get('name', 'unknown')[:40]}".replace(" ", "_"),
                            name=fdict.get("name", ""),
                            description=fdict.get("description", ""),
                            parent_product="routes",
                            entry_points=[],
                            business_capability_hint=fdict.get("business_capability_hint"),
                            confidence_score=fdict.get("confidence_score", 0.0),
                        ))
                    except (ValidationError, Exception):
                        pass
                existing_names = {f.name.lower() for f in final_features}
                final_features.extend(f for f in route_features if f.name.lower() not in existing_names)
            except Exception as exc:
                logger.warning("Route feature extraction failed: %s", exc)

        return ExtractionResult(
            source=source,
            features=final_features,
            total_clusters=total_clusters,
            skipped_clusters=skipped_clusters,
        )

    def _deduplicate(
        self, raw_features: list[dict], skipped_clusters: int, system_content: str = SYSTEM_PROMPT
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
            {"role": "system", "content": system_content},
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
