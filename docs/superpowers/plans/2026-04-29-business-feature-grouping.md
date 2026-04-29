# Business Feature Grouping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a post-extraction LLM pass that groups `DigitalFeature` objects into `BusinessFeature` clusters, each with a name, description, and list of constituent digital features.

**Architecture:** New `BusinessFeature` + `GroupingResult` Pydantic models → new `GroupingPrompts` → new `FeatureGrouper` class that takes an `ExtractionResult` and returns a `GroupingResult` → wired into CLI as a second pass after extraction → outputs added to `features.json`, `report.html`, `graph.html`.

**Tech Stack:** Pydantic v2, OpenAI-compatible LLM client (same as extraction), Click, Jinja2, pyvis.

---

### Task 1: Add Pydantic models for BusinessFeature and GroupingResult

**Files:**
- Modify: `src/extraction/models.py`
- Modify: `tests/extraction/test_models.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/extraction/test_models.py — add these tests

def test_business_feature_valid():
    from src.extraction.models import BusinessFeature, DigitalFeature, FeatureStatus
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
    with pytest.raises(Exception):
        BusinessFeature(id="x", name="x", description="x", digital_features=[], unknown="bad")

def test_grouping_result_valid():
    from src.extraction.models import BusinessFeature, GroupingResult
    gr = GroupingResult(
        source="./repo",
        business_features=[],
        ungrouped_feature_ids=[],
    )
    assert gr.source == "./repo"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/20011425/Dev/digitalFeaturesExtractor && source .venv/bin/activate
pytest tests/extraction/test_models.py -k "business_feature or grouping_result" -v
```
Expected: FAIL with `ImportError` or `cannot import name 'BusinessFeature'`

- [ ] **Step 3: Add models to `src/extraction/models.py`**

Append after `ExtractionResult`:

```python
class BusinessFeature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str
    digital_features: list[DigitalFeature]


class GroupingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    business_features: list[BusinessFeature]
    ungrouped_feature_ids: list[str]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/extraction/test_models.py -v
```
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/extraction/models.py tests/extraction/test_models.py
git commit -m "feat: add BusinessFeature and GroupingResult models"
```

---

### Task 2: Add grouping prompts

**Files:**
- Create: `src/extraction/grouping_prompts.py`
- Create: `tests/extraction/test_grouping_prompts.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/extraction/test_grouping_prompts.py
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
    raw = json.dumps([
        {
            "name": "Spot Discovery",
            "description": "Finding and exploring sport spots",
            "feature_ids": ["f1", "f2"]
        }
    ])
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/extraction/test_grouping_prompts.py -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create `src/extraction/grouping_prompts.py`**

```python
from __future__ import annotations

import json

GROUPING_SYSTEM_PROMPT = """You are an expert product analyst grouping Digital Features into Business Features.

A Business Feature is a high-level user-facing capability that groups related Digital Features under one coherent product domain.

RULES:
- Each group must have 2+ Digital Features (do not create single-feature groups unless truly isolated)
- Group name should be short and domain-oriented (e.g. "Spot Discovery", "Community Reporting")
- Description: one sentence explaining what this capability enables for the user
- Every Digital Feature must appear in exactly one group
- Features that don't fit any group go in a catch-all "Other" group

Return ONLY a valid JSON array of group objects. No markdown, no explanation."""


def build_grouping_prompt(features: list[dict]) -> str:
    features_json = json.dumps(
        [{"id": f.get("id", ""), "name": f.get("name", ""), "hint": f.get("business_capability_hint", "")} for f in features],
        indent=2,
    )
    return (
        "Group the following Digital Features into Business Features.\n\n"
        f"Digital Features:\n{features_json}\n\n"
        "Return a JSON array where each element has:\n"
        '- "name": short business domain name\n'
        '- "description": one sentence describing this capability\n'
        '- "feature_ids": array of Digital Feature ids belonging to this group\n\n'
        "Every feature id must appear in exactly one group. Return ONLY the JSON array."
    )


def parse_grouping_response(response: str) -> list[dict]:
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Grouping response is not valid JSON. Parse error: {exc}\n"
            f"Response (first 500 chars): {response[:500]}"
        ) from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
    return parsed
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/extraction/test_grouping_prompts.py -v
```
Expected: ALL PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/extraction/grouping_prompts.py tests/extraction/test_grouping_prompts.py
git commit -m "feat: add grouping prompts for business feature clustering"
```

---

### Task 3: Add FeatureGrouper class

**Files:**
- Create: `src/extraction/grouper.py`
- Modify: `src/extraction/__init__.py`
- Create: `tests/extraction/test_grouper.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/extraction/test_grouper.py
import pytest
from unittest.mock import MagicMock
from src.extraction.grouper import FeatureGrouper
from src.extraction.models import DigitalFeature, ExtractionResult


def _make_feature(id: str, name: str, hint: str | None = None) -> DigitalFeature:
    return DigitalFeature(
        id=id, name=name, description="desc",
        parent_product="p", entry_points=[],
        confidence_score=0.8, business_capability_hint=hint,
    )


def _make_result(features: list[DigitalFeature]) -> ExtractionResult:
    return ExtractionResult(
        source="./repo", features=features,
        total_clusters=5, skipped_clusters=0,
    )


def _mock_llm(response_text: str):
    import json
    client = MagicMock()
    msg = MagicMock()
    msg.content = response_text
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    client.chat.completions.create.return_value = completion
    return client


def test_group_returns_grouping_result():
    import json
    features = [
        _make_feature("f1", "Find spots", "Spot Discovery"),
        _make_feature("f2", "View spot details", "Spot Discovery"),
        _make_feature("f3", "Submit report", "Community"),
        _make_feature("f4", "View reports", "Community"),
    ]
    result = _make_result(features)
    llm_resp = json.dumps([
        {"name": "Spot Discovery", "description": "Find spots", "feature_ids": ["f1", "f2"]},
        {"name": "Community", "description": "Reports", "feature_ids": ["f3", "f4"]},
    ])
    grouper = FeatureGrouper(llm_client=_mock_llm(llm_resp), model="gpt-4.1")
    grouping = grouper.group(result)
    assert len(grouping.business_features) == 2
    assert grouping.source == "./repo"


def test_group_ungrouped_feature_ids():
    import json
    features = [
        _make_feature("f1", "Find spots"),
        _make_feature("f2", "Other thing"),
    ]
    result = _make_result(features)
    llm_resp = json.dumps([
        {"name": "Spot Discovery", "description": "desc", "feature_ids": ["f1"]},
    ])
    grouper = FeatureGrouper(llm_client=_mock_llm(llm_resp), model="gpt-4.1")
    grouping = grouper.group(result)
    assert "f2" in grouping.ungrouped_feature_ids


def test_group_empty_features():
    grouper = FeatureGrouper(llm_client=MagicMock(), model="gpt-4.1")
    result = _make_result([])
    grouping = grouper.group(result)
    assert grouping.business_features == []
    assert grouping.ungrouped_feature_ids == []


def test_group_llm_failure_returns_hint_based_groups():
    features = [
        _make_feature("f1", "Find spots", "Spot Discovery"),
        _make_feature("f2", "View spot", "Spot Discovery"),
        _make_feature("f3", "Submit report", "Community"),
    ]
    result = _make_result(features)
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("LLM down")
    grouper = FeatureGrouper(llm_client=client, model="gpt-4.1")
    grouping = grouper.group(result)
    assert len(grouping.business_features) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/extraction/test_grouper.py -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create `src/extraction/grouper.py`**

```python
from __future__ import annotations

import logging
from typing import Any

from .grouping_prompts import (
    GROUPING_SYSTEM_PROMPT,
    build_grouping_prompt,
    parse_grouping_response,
)
from .models import BusinessFeature, DigitalFeature, ExtractionResult, GroupingResult

logger = logging.getLogger(__name__)


class FeatureGrouper:
    def __init__(self, llm_client: Any, model: str = "gpt-4.1") -> None:
        self.llm_client = llm_client
        self.model = model

    def group(self, result: ExtractionResult) -> GroupingResult:
        if not result.features:
            return GroupingResult(
                source=result.source,
                business_features=[],
                ungrouped_feature_ids=[],
            )

        feature_dicts = [
            {
                "id": f.id,
                "name": f.name,
                "description": f.description,
                "business_capability_hint": f.business_capability_hint,
            }
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
            business_features.append(
                BusinessFeature(
                    id=f"bf_{i}_{group.get('name', 'group')[:30]}".replace(" ", "_"),
                    name=group.get("name", f"Group {i}"),
                    description=group.get("description", ""),
                    digital_features=matched,
                )
            )

        ungrouped = [f.id for f in result.features if f.id not in assigned_ids]

        return GroupingResult(
            source=result.source,
            business_features=business_features,
            ungrouped_feature_ids=ungrouped,
        )

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
        return GroupingResult(
            source=result.source,
            business_features=business_features,
            ungrouped_feature_ids=[],
        )
```

- [ ] **Step 4: Update `src/extraction/__init__.py`** to export `FeatureGrouper`, `BusinessFeature`, `GroupingResult`

```python
from .extractor import FeatureExtractor
from .grouper import FeatureGrouper
from .models import (
    BusinessFeature,
    DigitalFeature,
    ExtractionResult,
    FeatureStatus,
    GroupingResult,
)

__all__ = [
    "FeatureExtractor",
    "FeatureGrouper",
    "BusinessFeature",
    "DigitalFeature",
    "ExtractionResult",
    "FeatureStatus",
    "GroupingResult",
]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/extraction/test_grouper.py -v
```
Expected: ALL PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/extraction/grouper.py src/extraction/__init__.py tests/extraction/test_grouper.py
git commit -m "feat: add FeatureGrouper with LLM and hint-based fallback"
```

---

### Task 4: Update JSON exporter to include business features

**Files:**
- Modify: `src/output/json_exporter.py`
- Modify: `tests/output/test_json_exporter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/output/test_json_exporter.py — add this test

def test_export_with_grouping_result(tmp_path):
    import json
    from src.extraction.models import (
        BusinessFeature, DigitalFeature, ExtractionResult, GroupingResult
    )
    from src.output.json_exporter import JsonExporter

    f1 = DigitalFeature(
        id="f1", name="Find spots", description="desc",
        parent_product="p", entry_points=[], confidence_score=0.9
    )
    result = ExtractionResult(source="./repo", features=[f1], total_clusters=2, skipped_clusters=0)
    bf = BusinessFeature(id="bf1", name="Spot Discovery", description="desc", digital_features=[f1])
    grouping = GroupingResult(source="./repo", business_features=[bf], ungrouped_feature_ids=[])

    out = tmp_path / "features.json"
    JsonExporter().export(result, out, grouping=grouping)

    data = json.loads(out.read_text())
    assert "business_features" in data
    assert data["business_features"][0]["name"] == "Spot Discovery"
    assert len(data["business_features"][0]["digital_features"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/output/test_json_exporter.py::test_export_with_grouping_result -v
```
Expected: FAIL (wrong signature)

- [ ] **Step 3: Update `src/output/json_exporter.py`**

Change `export` signature to accept optional `grouping`:

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.extraction.models import ExtractionResult, GroupingResult


class JsonExporter:
    def export(
        self,
        result: ExtractionResult,
        output_path: Path,
        grouping: GroupingResult | None = None,
    ) -> None:
        sorted_features = sorted(result.features, key=lambda f: f.confidence_score, reverse=True)

        payload: dict = {
            "metadata": {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "total_features": len(result.features),
                "total_clusters": result.total_clusters,
                "skipped_clusters": result.skipped_clusters,
            },
            "features": [f.model_dump() for f in sorted_features],
        }

        if grouping is not None:
            payload["business_features"] = [
                {
                    "id": bf.id,
                    "name": bf.name,
                    "description": bf.description,
                    "digital_features": [f.model_dump() for f in bf.digital_features],
                }
                for bf in grouping.business_features
            ]
            if grouping.ungrouped_feature_ids:
                payload["ungrouped_feature_ids"] = grouping.ungrouped_feature_ids

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run all output tests**

```bash
pytest tests/output/test_json_exporter.py -v
```
Expected: ALL PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/output/json_exporter.py tests/output/test_json_exporter.py
git commit -m "feat: include business_features in JSON export"
```

---

### Task 5: Update HTML report to display business feature groups

**Files:**
- Modify: `templates/report.html.j2`
- Modify: `src/output/html_reporter.py`
- Modify: `tests/output/test_html_reporter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/output/test_html_reporter.py — add this test

def test_report_shows_business_features(tmp_path):
    from src.extraction.models import (
        BusinessFeature, DigitalFeature, ExtractionResult, GroupingResult
    )
    from src.output.html_reporter import HtmlReporter

    f1 = DigitalFeature(
        id="f1", name="Find spots", description="User can find nearby spots",
        parent_product="p", entry_points=[], confidence_score=0.9
    )
    result = ExtractionResult(source="./repo", features=[f1], total_clusters=2, skipped_clusters=0)
    bf = BusinessFeature(id="bf1", name="Spot Discovery", description="Explore sport spots", digital_features=[f1])
    grouping = GroupingResult(source="./repo", business_features=[bf], ungrouped_feature_ids=[])

    out = tmp_path / "report.html"
    HtmlReporter().export(result, out, source="./repo", grouping=grouping)

    html = out.read_text()
    assert "Spot Discovery" in html
    assert "Explore sport spots" in html
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/output/test_html_reporter.py::test_report_shows_business_features -v
```
Expected: FAIL

- [ ] **Step 3: Update `src/output/html_reporter.py`**

Add `grouping` parameter to `export`:

```python
def export(
    self,
    result: ExtractionResult,
    output_path: Path,
    source: str = "",
    grouping: GroupingResult | None = None,
) -> None:
    template = self.env.get_template("report.html.j2")
    html = template.render(result=result, source=source, grouping=grouping)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
```

Add the import at top: `from src.extraction.models import ExtractionResult, GroupingResult`

- [ ] **Step 4: Update `templates/report.html.j2`**

Add a "Business Features" section before the individual feature cards. Insert after the summary stats block and before the individual features grid:

```html
{% if grouping and grouping.business_features %}
<section class="business-features">
  <h2>Business Features</h2>
  <p class="subtitle">{{ grouping.business_features|length }} capability groups identified</p>
  {% for bf in grouping.business_features %}
  <div class="bf-card">
    <div class="bf-header">
      <h3>{{ bf.name }}</h3>
      <span class="bf-count">{{ bf.digital_features|length }} features</span>
    </div>
    <p class="bf-description">{{ bf.description }}</p>
    <ul class="bf-features">
      {% for f in bf.digital_features %}
      <li>
        <span class="dot" style="background: {{ 'var(--green)' if f.confidence_score >= 0.7 else 'var(--amber)' }}"></span>
        {{ f.name }}
        <span class="score">{{ (f.confidence_score * 100)|int }}%</span>
      </li>
      {% endfor %}
    </ul>
  </div>
  {% endfor %}
</section>
{% endif %}
```

Add CSS for `.business-features`, `.bf-card`, `.bf-header`, `.bf-count`, `.bf-description`, `.bf-features`, `.dot`, `.score` in the `<style>` block. Style similar to existing feature cards but with a left accent border per group.

- [ ] **Step 5: Run all HTML tests**

```bash
pytest tests/output/test_html_reporter.py -v
```
Expected: ALL PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add templates/report.html.j2 src/output/html_reporter.py tests/output/test_html_reporter.py
git commit -m "feat: display business feature groups in HTML report"
```

---

### Task 6: Wire FeatureGrouper into the CLI

**Files:**
- Modify: `src/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py — add this test

def test_analyze_calls_feature_grouper(tmp_path):
    from click.testing import CliRunner
    from unittest.mock import MagicMock, patch
    from src.cli import cli

    runner = CliRunner()
    with runner.isolated_filesystem():
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        (source_dir / "main.py").write_text("def hello(): pass")

        with patch("src.cli.LocalIngester") as mock_ing, \
             patch("src.cli.GraphifyWrapper") as mock_gw, \
             patch("src.cli.FeatureExtractor") as mock_fe, \
             patch("src.cli.FeatureGrouper") as mock_fg, \
             patch("src.cli.JsonExporter") as mock_je, \
             patch("src.cli.HtmlReporter") as mock_hr, \
             patch("src.cli.GraphVisualizer") as mock_gv, \
             patch("openai.OpenAI"):

            mock_ing.return_value.ingest.return_value = [source_dir / "main.py"]
            mock_gw.return_value.build_graph.return_value = MagicMock()
            mock_gw.return_value.get_clusters.return_value = {}
            mock_fe.return_value.extract.return_value = MagicMock(features=[], total_clusters=0, skipped_clusters=0)
            mock_fg.return_value.group.return_value = MagicMock(business_features=[], ungrouped_feature_ids=[])

            result = runner.invoke(cli, [
                "analyze", "--source", str(source_dir),
                "--api-key", "test-key", "--output-dir", str(tmp_path / "out")
            ])
            assert result.exit_code == 0
            mock_fg.return_value.group.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_cli.py::test_analyze_calls_feature_grouper -v
```
Expected: FAIL

- [ ] **Step 3: Update `src/cli.py`**

Add import:
```python
from src.extraction import FeatureExtractor, FeatureGrouper
```

In `analyze()`, after `extractor.extract(...)`:
```python
grouper = FeatureGrouper(llm_client, model)
grouping = grouper.group(result)
```

Pass `grouping` to exporters:
```python
JsonExporter().export(result, out / "features.json", grouping=grouping)
HtmlReporter().export(result, out / "report.html", source=source, grouping=grouping)
GraphVisualizer().export(graph, result.features, out / "graph.html")
```

Update summary line:
```python
click.echo(
    f"✓ Found {len(result.features)} features in {result.total_clusters} clusters"
    f" → {len(grouping.business_features)} business feature groups"
)
```

- [ ] **Step 4: Run all tests**

```bash
pytest -v
```
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/cli.py tests/test_cli.py
git commit -m "feat: wire FeatureGrouper into CLI analyze command"
```

---

### Task 7: Push to GitHub

- [ ] **Step 1: Verify clean state**

```bash
cd /Users/20011425/Dev/digitalFeaturesExtractor && source .venv/bin/activate
pytest -v
git status
```
Expected: all tests pass, working directory clean.

- [ ] **Step 2: Push**

```bash
git push origin main
```

- [ ] **Step 3: Run full analysis on Zephyr to smoke-test**

```bash
rm -rf /Users/20011425/Dev/Zephyr/dfe-cache
dfe analyze \
  --source /Users/20011425/Dev/Zephyr \
  --output-dir /Users/20011425/Dev/Zephyr/dfe-output \
  --api-key $OPENROUTER_API_KEY \
  --base-url https://openrouter.ai/api/v1 \
  --model anthropic/claude-sonnet-4-5 \
  --cache-dir /Users/20011425/Dev/Zephyr/dfe-cache \
  --verbose
```
Expected: output ends with `→ N business feature groups`

```bash
open /Users/20011425/Dev/Zephyr/dfe-output/report.html
```
