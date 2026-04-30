import json
from datetime import datetime, timezone
from pathlib import Path

from src.extraction.models import ExtractionResult, GroupingResult


class JsonExporter:
    def export(self, result: ExtractionResult, output_path: Path, grouping: GroupingResult | None = None) -> None:
        sorted_features = sorted(
            result.features, key=lambda f: f.confidence_score, reverse=True
        )
        payload = {
            "metadata": {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "total_features": len(result.features),
                "total_clusters": result.total_clusters,
                "skipped_clusters": result.skipped_clusters,
            },
            "features": [f.model_dump() for f in sorted_features],
        }
        if result.project_summary:
            payload["project_summary"] = result.project_summary
        if grouping is not None:
            payload["business_features"] = [
                {"id": bf.id, "name": bf.name, "description": bf.description,
                 "digital_features": [f.model_dump() for f in bf.digital_features]}
                for bf in grouping.business_features
            ]
            if grouping.ungrouped_feature_ids:
                payload["ungrouped_feature_ids"] = grouping.ungrouped_feature_ids
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
