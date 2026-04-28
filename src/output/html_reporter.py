from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.extraction.models import ExtractionResult, FeatureStatus

_DEFAULT_TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"


class HtmlReporter:
    def __init__(self, template_dir: Path | None = None) -> None:
        tdir = template_dir if template_dir is not None else _DEFAULT_TEMPLATE_DIR
        self._env = Environment(
            loader=FileSystemLoader(str(tdir)),
            autoescape=True,
        )

    def export(
        self,
        result: ExtractionResult,
        output_path: Path,
        source: str = "",
    ) -> None:
        template = self._env.get_template("report.html.j2")

        sorted_features = sorted(
            result.features, key=lambda f: f.confidence_score, reverse=True
        )

        avg_confidence = (
            f"{sum(f.confidence_score for f in result.features) / len(result.features):.0%}"
            if result.features
            else "N/A"
        )

        status_counts: dict[str, int] = {}
        for status in FeatureStatus:
            count = sum(1 for f in result.features if f.status == status)
            if count:
                status_counts[status.value] = count

        exported_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        html = template.render(
            features=sorted_features,
            source=source or result.source,
            exported_at=exported_at,
            avg_confidence=avg_confidence,
            status_counts=status_counts,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
