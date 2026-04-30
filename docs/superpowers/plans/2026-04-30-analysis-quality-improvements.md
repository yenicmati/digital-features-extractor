# Analysis Quality Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve feature extraction quality through better graph construction (TS/JS/Vue imports), richer LLM context (project description, Vue SFC parsing, route detection), and a two-pass filtering approach.

**Architecture:** Five independent improvements stacked on the existing pipeline. Each task is self-contained and leaves the test suite green. The order matters: Task 1 (import graph) feeds into Task 2 (Vue SFC); Tasks 3–5 are independent.

**Tech Stack:** Python 3.13, NetworkX, regex (stdlib), pathlib, existing LLM client (OpenAI-compatible)

---

## File Map

| File | What changes |
|---|---|
| `src/graph/wrapper.py` | Add TS/JS/Vue import parsing in `_fallback_extract` |
| `src/extraction/extractor.py` | Vue SFC content extractor; two-pass pre-filter; accept project context |
| `src/extraction/prompts.py` | `build_project_context_block`, `build_prefilter_prompt` |
| `src/cli.py` | Read README + package.json; detect router files; pass to extractor |
| `tests/graph/test_wrapper.py` | Tests for TS/JS/Vue import edges |
| `tests/extraction/test_extractor.py` | Tests for Vue SFC extraction, two-pass, project context |
| `tests/extraction/test_prompts.py` | Tests for new prompt builders |

---

## Task 1 — TS/JS/Vue import graph

**Goal:** Create edges between files when one imports another (`.ts`, `.js`, `.tsx`, `.jsx`, `.vue`). Currently only Python imports are resolved, leaving all JS/TS/Vue files as isolated nodes.

**Files:**
- Modify: `src/graph/wrapper.py` — `_fallback_extract` function
- Modify: `tests/graph/test_wrapper.py`

### Step 1 — Write failing tests

```python
# tests/graph/test_wrapper.py  — add after existing tests

import tempfile, shutil
from pathlib import Path
import pytest
from src.graph.wrapper import GraphifyWrapper


@pytest.fixture()
def ts_repo(tmp_path):
    (tmp_path / "Dashboard.vue").write_text(
        "<script setup>\nimport { useMilestones } from './useMilestones'\n</script>\n<template><div/></template>"
    )
    (tmp_path / "useMilestones.ts").write_text(
        "export function useMilestones() { return [] }"
    )
    (tmp_path / "utils.ts").write_text("export const PI = 3.14")
    return tmp_path


def test_ts_import_creates_edge(ts_repo):
    files = list(ts_repo.glob("*"))
    wrapper = GraphifyWrapper()
    graph = wrapper.build_graph(files)
    # Dashboard.vue imports useMilestones.ts → edge must exist
    node_names = {attrs.get("name", n) for n, attrs in graph.nodes(data=True)}
    assert any("Dashboard" in n or "dashboard" in n.lower() for n in graph.nodes())
    assert any("useMilestones" in n or "usemilestones" in n.lower() for n in graph.nodes())
    # There should be at least one import edge
    edge_types = [data.get("type") for _, _, data in graph.edges(data=True)]
    assert "imports" in edge_types


def test_vue_without_imports_has_no_import_edges(tmp_path):
    (tmp_path / "Standalone.vue").write_text(
        "<template><div>Hello</div></template>"
    )
    files = list(tmp_path.glob("*.vue"))
    wrapper = GraphifyWrapper()
    graph = wrapper.build_graph(files)
    edge_types = [data.get("type") for _, _, data in graph.edges(data=True)]
    assert "imports" not in edge_types


def test_js_import_creates_edge(tmp_path):
    (tmp_path / "app.js").write_text("import { helper } from './helper'")
    (tmp_path / "helper.js").write_text("export function helper() {}")
    files = list(tmp_path.glob("*.js"))
    wrapper = GraphifyWrapper()
    graph = wrapper.build_graph(files)
    edge_types = [data.get("type") for _, _, data in graph.edges(data=True)]
    assert "imports" in edge_types
```

- [ ] **Step 1:** Add the three test functions above to `tests/graph/test_wrapper.py`

- [ ] **Step 2:** Run tests to confirm they fail
```
cd /Users/20011425/Dev/digitalFeaturesExtractor && source .venv/bin/activate
pytest tests/graph/test_wrapper.py::test_ts_import_creates_edge tests/graph/test_wrapper.py::test_js_import_creates_edge -v
```
Expected: FAIL

- [ ] **Step 3:** Implement TS/JS/Vue import parsing in `_fallback_extract`

Replace the loop starting at `for f in files:` / `if f.suffix != ".py":` block with:

```python
import re as _re

_TS_IMPORT_RE = _re.compile(
    r"""(?:import|export)\s+.*?from\s+['"](\./[^'"]+)['"]""",
    _re.DOTALL,
)

def _resolve_ts_import(raw: str, files: list[Path]) -> Path | None:
    stem = raw.lstrip("./").split("/")[-1]
    stem_no_ext = stem.rsplit(".", 1)[0] if "." in stem else stem
    return next(
        (p for p in files if p.stem.lower() == stem_no_ext.lower()),
        None,
    )

def _extract_script_block(text: str) -> str:
    m = _re.search(r"<script[^>]*>(.*?)</script>", text, _re.DOTALL | _re.IGNORECASE)
    return m.group(1) if m else text
```

Then inside `_fallback_extract`, replace:

```python
for f in files:
    if f.suffix != ".py":
        continue
    try:
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
    except SyntaxError:
        continue
    fid = file_ids[f]
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            nid = f"{fid}_{node.name.lower()}"
            ntype = "class" if isinstance(node, ast.ClassDef) else "function"
            nodes.append({"id": nid, "type": ntype, "name": node.name, "source_file": str(f)})
            edges.append({"source": fid, "target": nid, "type": "contains"})
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else ([node.module] if node.module else [])
            )
            for name in names:
                mod_stem = name.split(".")[-1].lower()
                target_file = next(
                    (p for p in files if p.stem.lower() == mod_stem), None
                )
                if target_file:
                    edges.append(
                        {"source": fid, "target": file_ids[target_file], "type": "imports"}
                    )
```

with:

```python
_TS_IMPORT_RE = re.compile(
    r"""(?:import|export)\s+.*?from\s+['"](\./[^'"]+)['"]""",
    re.DOTALL,
)

def _resolve_ts_import(raw: str, all_files: list[Path]) -> Path | None:
    stem = raw.lstrip("./").split("/")[-1]
    stem_no_ext = stem.rsplit(".", 1)[0] if "." in stem else stem
    return next((p for p in all_files if p.stem.lower() == stem_no_ext.lower()), None)

def _extract_script_block(text: str) -> str:
    m = re.search(r"<script[^>]*>(.*?)</script>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else text

for f in files:
    fid = file_ids[f]
    try:
        text = f.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue

    if f.suffix == ".py":
        try:
            tree = ast.parse(text, filename=str(f))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                nid = f"{fid}_{node.name.lower()}"
                ntype = "class" if isinstance(node, ast.ClassDef) else "function"
                nodes.append({"id": nid, "type": ntype, "name": node.name, "source_file": str(f)})
                edges.append({"source": fid, "target": nid, "type": "contains"})
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else ([node.module] if node.module else [])
                )
                for name in names:
                    mod_stem = name.split(".")[-1].lower()
                    target_file = next((p for p in files if p.stem.lower() == mod_stem), None)
                    if target_file:
                        edges.append({"source": fid, "target": file_ids[target_file], "type": "imports"})

    elif f.suffix in {".ts", ".tsx", ".js", ".jsx", ".vue"}:
        body = _extract_script_block(text) if f.suffix == ".vue" else text
        for match in _TS_IMPORT_RE.finditer(body):
            target_file = _resolve_ts_import(match.group(1), files)
            if target_file and target_file != f:
                edges.append({"source": fid, "target": file_ids[target_file], "type": "imports"})
```

Note: add `import re` at the top of `wrapper.py` (it's not currently imported).

- [ ] **Step 4:** Run the new tests
```
pytest tests/graph/test_wrapper.py -v
```
Expected: all PASS

- [ ] **Step 5:** Run full suite
```
pytest -q
```
Expected: all pass

- [ ] **Step 6:** Commit
```bash
git add src/graph/wrapper.py tests/graph/test_wrapper.py
git commit -m "feat: parse TS/JS/Vue imports to build cross-file graph edges"
```

---

## Task 2 — Vue SFC content extraction

**Goal:** For `.vue` files, extract `<template>` and `<script>` content separately rather than dumping the first 60 lines. The template HTML is the most user-facing signal.

**Files:**
- Modify: `src/extraction/extractor.py` — add `_extract_file_content` helper, use it in `extract()`
- Modify: `tests/extraction/test_extractor.py`

### Step 1 — Write failing tests

```python
# tests/extraction/test_extractor.py — add at end

def test_vue_sfc_content_prioritises_template(tmp_path):
    vue_file = tmp_path / "MilestoneCard.vue"
    vue_file.write_text(
        "<template>\n  <div class='card'>Milestone: {{ name }}</div>\n</template>\n"
        "<script setup>\nconst props = defineProps(['name'])\n</script>\n"
    )
    extractor = FeatureExtractor(llm_client=MagicMock(), model="gpt-4o")
    content = extractor._extract_file_content(vue_file)
    assert "Milestone" in content
    assert "[template]" in content


def test_non_vue_content_returns_first_60_lines(tmp_path):
    ts_file = tmp_path / "service.ts"
    lines = [f"line {i}" for i in range(100)]
    ts_file.write_text("\n".join(lines))
    extractor = FeatureExtractor(llm_client=MagicMock(), model="gpt-4o")
    content = extractor._extract_file_content(ts_file)
    assert "line 0" in content
    assert "line 59" in content
    assert "line 60" not in content
```

- [ ] **Step 1:** Add the two test functions above to `tests/extraction/test_extractor.py`

- [ ] **Step 2:** Run tests to confirm they fail
```
pytest tests/extraction/test_extractor.py::test_vue_sfc_content_prioritises_template tests/extraction/test_extractor.py::test_non_vue_content_returns_first_60_lines -v
```
Expected: FAIL (AttributeError: no `_extract_file_content`)

- [ ] **Step 3:** Add `_extract_file_content` method to `FeatureExtractor` in `src/extraction/extractor.py`

Add `import re` at the top. Then add this method to the class, before `extract()`:

```python
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
```

- [ ] **Step 4:** Replace the inline content-reading block in `extract()` with a call to `_extract_file_content`

In `extract()`, replace:

```python
                content: str | None = None
                if path_str:
                    try:
                        file_path = Path(path_str)
                        if file_path.exists():
                            raw_lines = file_path.read_text(
                                encoding="utf-8", errors="ignore"
                            ).splitlines()[:60]
                            content = "\n".join(raw_lines)
                    except Exception:
                        pass
```

with:

```python
                content: str | None = None
                if path_str:
                    fp = Path(path_str)
                    if fp.exists():
                        content = self._extract_file_content(fp)
```

- [ ] **Step 5:** Run the new tests
```
pytest tests/extraction/test_extractor.py -v
```
Expected: all PASS

- [ ] **Step 6:** Full suite
```
pytest -q
```

- [ ] **Step 7:** Commit
```bash
git add src/extraction/extractor.py tests/extraction/test_extractor.py
git commit -m "feat: extract Vue SFC template/script sections for richer LLM context"
```

---

## Task 3 — Project context injection (README + package.json)

**Goal:** Pass the project name, description, and a README excerpt to the LLM before cluster analysis, so it understands the domain.

**Files:**
- Modify: `src/extraction/prompts.py` — add `build_project_context_block`
- Modify: `src/extraction/extractor.py` — accept `project_context` param, inject into messages
- Modify: `src/cli.py` — detect README + package.json, extract context, pass to extractor
- Modify: `tests/extraction/test_prompts.py`, `tests/extraction/test_extractor.py`

### Step 1 — Write failing tests

```python
# tests/extraction/test_prompts.py — add at end

from src.extraction.prompts import build_project_context_block

def test_build_project_context_block_contains_name_and_description():
    block = build_project_context_block("debt-viewer", "Tool for visualizing technical debt", "## Overview\nThis app shows...")
    assert "debt-viewer" in block
    assert "technical debt" in block

def test_build_project_context_block_truncates_readme():
    long_readme = "x" * 2000
    block = build_project_context_block("app", "desc", long_readme)
    assert len(block) < 2000


# tests/extraction/test_extractor.py — add at end

def test_project_context_injected_into_messages():
    client = make_llm_client(VALID_RESPONSE)
    extractor = FeatureExtractor(llm_client=client, model="gpt-4o")
    extractor.extract(
        make_clusters(),
        make_graph(),
        source="test",
        project_context="This is a debt visualization tool",
    )
    calls = client.chat.completions.create.call_args_list
    all_messages = [msg for call in calls for msg in call.kwargs.get("messages", [])]
    all_content = " ".join(m["content"] for m in all_messages)
    assert "debt visualization" in all_content
```

- [ ] **Step 1:** Add the tests above to the respective test files

- [ ] **Step 2:** Run to confirm FAIL
```
pytest tests/extraction/test_prompts.py::test_build_project_context_block_contains_name_and_description tests/extraction/test_extractor.py::test_project_context_injected_into_messages -v
```

- [ ] **Step 3:** Add `build_project_context_block` to `src/extraction/prompts.py`

```python
def build_project_context_block(
    name: str,
    description: str,
    readme_excerpt: str,
) -> str:
    readme_trimmed = readme_excerpt.strip()[:600]
    return (
        f"[Project context]\n"
        f"Name: {name}\n"
        f"Description: {description}\n"
        f"README excerpt:\n{readme_trimmed}"
    )
```

- [ ] **Step 4:** Update `FeatureExtractor.extract()` signature and inject context

In `src/extraction/extractor.py`, update the `extract` method signature:

```python
def extract(
    self,
    clusters: dict[str, list[str]],
    graph: nx.Graph,
    source: str = "unknown",
    project_context: str | None = None,
) -> ExtractionResult:
```

At the top of the method body (before the cluster loop), build the system message:

```python
        system_content = SYSTEM_PROMPT
        if project_context:
            system_content = f"{SYSTEM_PROMPT}\n\n{project_context}"
```

Then replace every `{"role": "system", "content": SYSTEM_PROMPT}` inside `extract()` with `{"role": "system", "content": system_content}`.

Also update `_deduplicate` to accept and use `system_content`:

Change signature to:
```python
def _deduplicate(self, raw_features: list[dict], skipped_clusters: int, system_content: str = SYSTEM_PROMPT) -> list[DigitalFeature]:
```

Update the messages list inside `_deduplicate`:
```python
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": summary_prompt},
        ]
```

And update the call site in `extract()`:
```python
        final_features = self._deduplicate(all_raw_features, skipped_clusters, system_content)
```

- [ ] **Step 5:** Update `src/cli.py` to detect README + package.json

Add a helper function after the existing `_build_llm_client`:

```python
def _detect_project_context(root: Path) -> str | None:
    import json as _json

    name: str | None = None
    description: str | None = None
    readme_text: str = ""

    pkg = root / "package.json"
    if pkg.exists():
        try:
            data = _json.loads(pkg.read_text(encoding="utf-8", errors="ignore"))
            name = data.get("name")
            description = data.get("description")
        except Exception:
            pass

    for readme_name in ("README.md", "readme.md", "README.rst", "README"):
        readme_file = root / readme_name
        if readme_file.exists():
            readme_text = readme_file.read_text(encoding="utf-8", errors="ignore")
            break

    if not name and not description and not readme_text:
        return None

    from src.extraction.prompts import build_project_context_block
    return build_project_context_block(
        name or root.name,
        description or "",
        readme_text,
    )
```

In `analyze()`, after `files = ingester.ingest(source)`, detect the root dir and build context:

```python
    # Detect project root — for GitHub repos it's the temp clone dir; for local it's the source path
    if _is_github_source(source):
        from src.ingestion.github_ingester import GithubIngester as _GHI
        project_root = Path(ingester.temp_dir) if hasattr(ingester, "temp_dir") and ingester.temp_dir else None
    else:
        project_root = Path(source)
    project_context = _detect_project_context(project_root) if project_root else None
```

Then pass it to extractor:
```python
    result = extractor.extract(clusters, graph, source=source, project_context=project_context)
```

- [ ] **Step 6:** Run all new tests
```
pytest tests/extraction/test_prompts.py tests/extraction/test_extractor.py -v
```
Expected: all PASS

- [ ] **Step 7:** Full suite
```
pytest -q
```

- [ ] **Step 8:** Commit
```bash
git add src/extraction/prompts.py src/extraction/extractor.py src/cli.py tests/extraction/test_prompts.py tests/extraction/test_extractor.py
git commit -m "feat: inject README + package.json project context into LLM prompts"
```

---

## Task 4 — Route-based feature detection

**Goal:** Detect router files (vue-router, react-router, Angular routes), extract route definitions, and inject them as high-confidence feature candidates into the summary pass.

**Files:**
- Modify: `src/extraction/prompts.py` — add `build_routes_prompt`
- Modify: `src/extraction/extractor.py` — add `_extract_routes_from_files`, use in `extract()`
- Modify: `src/cli.py` — pass `files` to extractor for route detection
- Modify: `tests/extraction/test_extractor.py`, `tests/extraction/test_prompts.py`

### Step 1 — Write failing tests

```python
# tests/extraction/test_extractor.py — add at end

def test_router_file_routes_are_detected(tmp_path):
    router_file = tmp_path / "router.ts"
    router_file.write_text(
        "const routes = [\n"
        "  { path: '/dashboard', component: Dashboard, name: 'Dashboard' },\n"
        "  { path: '/reports', component: Reports, name: 'Reports' },\n"
        "]\n"
    )
    extractor = FeatureExtractor(llm_client=MagicMock(), model="gpt-4o")
    routes = extractor._extract_routes_from_files([router_file])
    assert len(routes) >= 2
    paths = [r["path"] for r in routes]
    assert "/dashboard" in paths
    assert "/reports" in paths


def test_no_routes_when_no_router_file(tmp_path):
    service_file = tmp_path / "service.ts"
    service_file.write_text("export class MyService {}")
    extractor = FeatureExtractor(llm_client=MagicMock(), model="gpt-4o")
    routes = extractor._extract_routes_from_files([service_file])
    assert routes == []


# tests/extraction/test_prompts.py — add at end

from src.extraction.prompts import build_routes_prompt

def test_build_routes_prompt_lists_paths():
    routes = [
        {"path": "/dashboard", "name": "Dashboard"},
        {"path": "/settings", "name": "Settings"},
    ]
    prompt = build_routes_prompt(routes)
    assert "/dashboard" in prompt
    assert "/settings" in prompt
```

- [ ] **Step 1:** Add tests above

- [ ] **Step 2:** Run to confirm FAIL
```
pytest tests/extraction/test_extractor.py::test_router_file_routes_are_detected tests/extraction/test_extractor.py::test_no_routes_when_no_router_file tests/extraction/test_prompts.py::test_build_routes_prompt_lists_paths -v
```

- [ ] **Step 3:** Add `build_routes_prompt` to `src/extraction/prompts.py`

```python
def build_routes_prompt(routes: list[dict]) -> str:
    route_lines = "\n".join(
        f"  - {r.get('path', '?')}  (name: {r.get('name', 'unknown')})"
        for r in routes
    )
    return (
        "The following application routes were detected. Each route likely corresponds to a "
        "user-facing screen or feature. Use them as high-confidence candidates when identifying "
        "Digital Features.\n\n"
        f"Routes:\n{route_lines}\n\n"
        "For each route that corresponds to a meaningful user capability, include it in the "
        "feature list with confidence ≥ 0.8 unless the name clearly indicates a technical or "
        "auth-only page (e.g. /callback, /health, /404)."
    )
```

- [ ] **Step 4:** Add `_extract_routes_from_files` to `FeatureExtractor` in `src/extraction/extractor.py`

Add `import re` if not already present. Add this method:

```python
_ROUTER_FILE_NAMES = frozenset({
    "router", "routes", "index", "app-routing.module",
    "app.router", "routing",
})
_ROUTE_PATH_RE = re.compile(r"""path\s*:\s*['"]([^'"]+)['"]""")
_ROUTE_NAME_RE = re.compile(r"""name\s*:\s*['"]([^'"]+)['"]""")

def _extract_routes_from_files(self, files: list[Path]) -> list[dict]:
    router_files = [
        f for f in files
        if f.stem.lower().replace("-", "").replace(".", "") in {
            n.replace("-", "").replace(".", "") for n in self._ROUTER_FILE_NAMES
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
```

- [ ] **Step 5:** Integrate routes into `extract()` method

Update `extract()` signature to accept `files`:

```python
def extract(
    self,
    clusters: dict[str, list[str]],
    graph: nx.Graph,
    source: str = "unknown",
    project_context: str | None = None,
    files: list[Path] | None = None,
) -> ExtractionResult:
```

After the cluster loop (before `_deduplicate`), add:

```python
        routes = self._extract_routes_from_files(files) if files else []
        if routes:
            routes_block = build_routes_prompt(routes)
            route_messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": routes_block},
            ]
            try:
                raw_text = self._get_llm_response("__routes__", route_messages)
                route_features = parse_llm_response(raw_text)
                for fdict in route_features:
                    all_raw_features.append({**fdict, "_parent_product": "routes"})
            except Exception as exc:
                logger.warning("Route feature extraction failed: %s", exc)
```

Also add to imports at top of file:
```python
from .prompts import (
    SYSTEM_PROMPT,
    build_cluster_prompt,
    build_routes_prompt,
    build_summary_prompt,
    parse_llm_response,
)
```

- [ ] **Step 6:** Update `src/cli.py` to pass files to extractor

```python
    result = extractor.extract(clusters, graph, source=source, project_context=project_context, files=files)
```

- [ ] **Step 7:** Run new tests
```
pytest tests/extraction/test_extractor.py tests/extraction/test_prompts.py -v
```
Expected: all PASS

- [ ] **Step 8:** Full suite
```
pytest -q
```

- [ ] **Step 9:** Commit
```bash
git add src/extraction/prompts.py src/extraction/extractor.py src/cli.py tests/extraction/test_extractor.py tests/extraction/test_prompts.py
git commit -m "feat: detect router files and extract routes as high-confidence feature candidates"
```

---

## Task 5 — Two-pass pre-filter (skip purely technical clusters)

**Goal:** Before expensive feature extraction, ask the LLM which clusters contain user-facing code worth analyzing. This reduces noise (utils, config, models get skipped) and cuts LLM call count on large repos.

**Files:**
- Modify: `src/extraction/prompts.py` — add `build_prefilter_prompt`
- Modify: `src/extraction/extractor.py` — add `_prefilter_clusters`, call in `extract()`
- Modify: `tests/extraction/test_extractor.py`, `tests/extraction/test_prompts.py`

### Step 1 — Write failing tests

```python
# tests/extraction/test_prompts.py — add at end

from src.extraction.prompts import build_prefilter_prompt

def test_build_prefilter_prompt_contains_cluster_ids():
    summaries = {"cluster_0": "Dashboard component showing metrics", "cluster_1": "HTTP utility functions"}
    prompt = build_prefilter_prompt(summaries)
    assert "cluster_0" in prompt
    assert "cluster_1" in prompt


# tests/extraction/test_extractor.py — add at end

def test_prefilter_keeps_user_facing_clusters():
    client = make_llm_client('["cluster_ui"]')
    extractor = FeatureExtractor(llm_client=client, model="gpt-4o")
    summaries = {
        "cluster_ui": "Dashboard showing quality metrics to users",
        "cluster_util": "HTTP retry utility functions",
    }
    kept = extractor._prefilter_clusters(summaries)
    assert "cluster_ui" in kept


def test_prefilter_falls_back_to_all_clusters_on_error():
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("network error")
    extractor = FeatureExtractor(llm_client=client, model="gpt-4o")
    summaries = {"cluster_a": "foo", "cluster_b": "bar"}
    kept = extractor._prefilter_clusters(summaries)
    assert set(kept) == {"cluster_a", "cluster_b"}
```

- [ ] **Step 1:** Add tests above

- [ ] **Step 2:** Run to confirm FAIL
```
pytest tests/extraction/test_extractor.py::test_prefilter_keeps_user_facing_clusters tests/extraction/test_extractor.py::test_prefilter_falls_back_to_all_clusters_on_error tests/extraction/test_prompts.py::test_build_prefilter_prompt_contains_cluster_ids -v
```

- [ ] **Step 3:** Add `build_prefilter_prompt` to `src/extraction/prompts.py`

```python
PREFILTER_SYSTEM_PROMPT = """You are filtering code clusters to find those containing user-facing features.

Return a JSON array of cluster_id strings — only clusters that contain UI components, pages, user interactions, or business logic visible to end users.

EXCLUDE clusters that are purely: HTTP utilities, config files, type definitions, test helpers, error handlers, API clients, database models, generic utilities.

Return ONLY a JSON array of strings. Example: ["cluster_2", "cluster_5"]"""


def build_prefilter_prompt(summaries: dict[str, str]) -> str:
    lines = "\n".join(f"  {cid}: {desc}" for cid, desc in summaries.items())
    return (
        "Given the following code clusters and their one-line descriptions, return the IDs of "
        "clusters that are worth analyzing for user-facing Digital Features.\n\n"
        f"Clusters:\n{lines}\n\n"
        "Return ONLY a JSON array of cluster_id strings to keep."
    )
```

- [ ] **Step 4:** Add `_prefilter_clusters` to `FeatureExtractor`

Add constant at class level:
```python
_PREFILTER_MIN_CLUSTERS = 6
```

Add method:
```python
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
```

- [ ] **Step 5:** Wire pre-filter into `extract()`

After the micro-cluster merging block (where `clusters_to_process` is built), before the main feature-extraction loop, add:

```python
        if len(clusters_to_process) >= self._PREFILTER_MIN_CLUSTERS:
            cluster_summaries = {
                cid: ", ".join(node_names[:5])
                for cid, node_names in clusters_to_process.items()
            }
            kept_ids = self._prefilter_clusters(cluster_summaries)
            clusters_to_process = {k: v for k, v in clusters_to_process.items() if k in kept_ids}
            logger.debug("Pre-filter: kept %d / %d clusters", len(clusters_to_process), len(cluster_summaries))
```

- [ ] **Step 6:** Add `PREFILTER_SYSTEM_PROMPT` and `build_prefilter_prompt` to the import in `extractor.py`

```python
from .prompts import (
    PREFILTER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_cluster_prompt,
    build_prefilter_prompt,
    build_routes_prompt,
    build_summary_prompt,
    parse_llm_response,
)
```

- [ ] **Step 7:** Run new tests
```
pytest tests/extraction/test_extractor.py tests/extraction/test_prompts.py -v
```
Expected: all PASS

- [ ] **Step 8:** Full suite
```
pytest -q
```

- [ ] **Step 9:** Commit
```bash
git add src/extraction/prompts.py src/extraction/extractor.py tests/extraction/test_extractor.py tests/extraction/test_prompts.py
git commit -m "feat: two-pass pre-filter to skip purely technical clusters before LLM feature extraction"
```

---

## Final validation

- [ ] **Run full suite one last time**
```
pytest -q
```
Expected: all pass (≥ 80 tests)

- [ ] **Re-run on adeo/s2e** to verify improvement
```bash
cd /Users/20011425/Dev/digitalFeaturesExtractor && source .venv/bin/activate
rm -rf /Users/20011425/Dev/dfe-adeo-s2e-cache /Users/20011425/Dev/dfe-adeo-s2e
dfe analyze \
  --source https://github.com/adeo/architecture-insights-package-s2e-trajectories \
  --output-dir /Users/20011425/Dev/dfe-adeo-s2e \
  --api-key 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 \
  --base-url https://openrouter.ai/api/v1 \
  --model anthropic/claude-sonnet-4-5 \
  --cache-dir /Users/20011425/Dev/dfe-adeo-s2e-cache \
  --verbose
```
Expected: more features than before (was 6), higher confidence scores.

- [ ] **Commit final + push**
```bash
git push
```
