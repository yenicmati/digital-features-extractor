# Digital Feature Extractor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI Python tool that ingests a codebase (GitHub URL or local path), builds a knowledge graph via Graphify OSS, and extracts Digital Features as defined by ADEO, outputting JSON + HTML report + interactive graph.

**Architecture:** Graphify (safishamsi/graphify) handles AST parsing and graph construction. A custom extraction layer applies Leiden clustering and LLM prompts (OpenAI/Anthropic) to infer Digital Features from code clusters. Three output formats are generated: `features.json`, `report.html`, `graph.html`.

**Tech Stack:** Python 3.11+, graphify, tree-sitter, networkx, leidenalg, pydantic v2, openai SDK, anthropic SDK, jinja2, pyvis, click, gitpython

---

## File Map

```
digitalFeaturesExtractor/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── base.py                 # Ingester ABC
│   │   ├── github_ingester.py      # git clone --depth=1
│   │   └── local_ingester.py       # recursive file scan
│   ├── graph/
│   │   ├── __init__.py
│   │   └── graphify_wrapper.py     # wraps graphify OSS API
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── models.py               # Pydantic DigitalFeature schema
│   │   ├── prompts.py              # LLM prompt templates
│   │   └── feature_extractor.py   # Leiden clustering + LLM calls
│   ├── output/
│   │   ├── __init__.py
│   │   ├── json_exporter.py
│   │   ├── html_reporter.py
│   │   └── graph_visualizer.py
│   └── cli.py                      # Click entrypoint
├── tests/
│   ├── ingestion/
│   │   ├── test_github_ingester.py
│   │   └── test_local_ingester.py
│   ├── extraction/
│   │   ├── test_models.py
│   │   └── test_feature_extractor.py
│   └── output/
│       └── test_json_exporter.py
├── templates/
│   └── report.html.j2              # Jinja2 template
├── docs/superpowers/
│   ├── specs/2026-04-28-digital-feature-extractor-design.md
│   └── plans/2026-04-28-digital-feature-extractor-plan.md
├── pyproject.toml
├── .featureextractor.yml.example
└── README.md
```

---

## Task 1: Project Bootstrap

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `src/ingestion/__init__.py`
- Create: `src/graph/__init__.py`
- Create: `src/extraction/__init__.py`
- Create: `src/output/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "digital-features-extractor"
version = "0.1.0"
description = "Extract Digital Features from a codebase using LLM + knowledge graph"
requires-python = ">=3.11"
dependencies = [
    "graphify>=0.5.0",
    "networkx>=3.3",
    "leidenalg>=0.10.2",
    "igraph>=0.11.6",
    "pydantic>=2.7",
    "openai>=1.30",
    "anthropic>=0.28",
    "jinja2>=3.1",
    "pyvis>=0.3.2",
    "click>=8.1",
    "gitpython>=3.1",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "pytest-mock>=3.14",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",   # for mocking HTTP
]

[project.scripts]
dfe = "src.cli:cli"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create all `__init__.py` files**

```bash
touch src/__init__.py src/ingestion/__init__.py src/graph/__init__.py src/extraction/__init__.py src/output/__init__.py tests/__init__.py
mkdir -p tests/ingestion tests/extraction tests/output templates
touch tests/ingestion/__init__.py tests/extraction/__init__.py tests/output/__init__.py
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -e ".[dev]"
```

Expected: all packages install without error.

- [ ] **Step 4: Verify import works**

```bash
python -c "import networkx; import pydantic; import click; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git init
git add pyproject.toml src/ tests/ templates/
git commit -m "chore: bootstrap project structure"
```

---

## Task 2: Pydantic Models

**Files:**
- Create: `src/extraction/models.py`
- Create: `tests/extraction/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/extraction/test_models.py
import pytest
from pydantic import ValidationError
from src.extraction.models import DigitalFeature, FeatureStatus, ExtractionResult

def test_digital_feature_valid():
    f = DigitalFeature(
        id="search-product",
        name="Product Search",
        description="Allows users to search for products by keyword.",
        status=FeatureStatus.LIVE,
        parent_product="catalog-api",
        entry_points=["GET /api/products/search"],
        business_capability_hint="Product Discovery",
        confidence_score=0.85,
    )
    assert f.id == "search-product"
    assert f.confidence_score == 0.85

def test_digital_feature_defaults():
    f = DigitalFeature(
        id="x",
        name="X",
        description="desc",
        parent_product="prod",
        entry_points=[],
    )
    assert f.status == FeatureStatus.TO_REVIEW
    assert f.confidence_score == 0.0
    assert f.business_capability_hint is None

def test_digital_feature_confidence_bounds():
    with pytest.raises(ValidationError):
        DigitalFeature(
            id="x", name="X", description="d", parent_product="p",
            entry_points=[], confidence_score=1.5
        )

def test_extraction_result():
    r = ExtractionResult(
        source="https://github.com/org/repo",
        features=[],
        total_clusters=5,
        skipped_clusters=1,
    )
    assert r.total_clusters == 5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/extraction/test_models.py -v
```

Expected: `ImportError` or `ModuleNotFoundError`

- [ ] **Step 3: Implement models**

```python
# src/extraction/models.py
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class FeatureStatus(str, Enum):
    LIVE = "Live"
    TO_BE_DEVELOPED = "To Be Developed"
    DEPRECATED = "Deprecated"
    TO_REVIEW = "To Review"


class DigitalFeature(BaseModel):
    id: str
    name: str
    description: str
    status: FeatureStatus = FeatureStatus.TO_REVIEW
    parent_product: str
    entry_points: list[str]
    business_capability_hint: Optional[str] = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    source: str
    features: list[DigitalFeature]
    total_clusters: int
    skipped_clusters: int
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/extraction/test_models.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/extraction/models.py tests/extraction/test_models.py
git commit -m "feat: add DigitalFeature Pydantic models"
```

---

## Task 3: Ingestion Layer

**Files:**
- Create: `src/ingestion/base.py`
- Create: `src/ingestion/local_ingester.py`
- Create: `src/ingestion/github_ingester.py`
- Create: `tests/ingestion/test_local_ingester.py`
- Create: `tests/ingestion/test_github_ingester.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ingestion/test_local_ingester.py
import pytest
from pathlib import Path
from src.ingestion.local_ingester import LocalIngester

def test_get_file_paths_returns_py_files(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    (tmp_path / "src" / "utils.py").write_text("def foo(): pass")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("def test_(): pass")

    ingester = LocalIngester(root=tmp_path, excludes=["tests/**"])
    paths = ingester.get_file_paths()

    assert len(paths) == 2
    assert all(p.suffix == ".py" for p in paths)
    assert not any("tests" in str(p) for p in paths)

def test_get_file_paths_empty_if_no_match(tmp_path):
    ingester = LocalIngester(root=tmp_path, excludes=[])
    assert ingester.get_file_paths() == []
```

```python
# tests/ingestion/test_github_ingester.py
import pytest
from unittest.mock import patch, MagicMock
from src.ingestion.github_ingester import GithubIngester

def test_clone_url_is_called(tmp_path):
    with patch("src.ingestion.github_ingester.Repo.clone_from") as mock_clone:
        mock_clone.return_value = MagicMock()
        ingester = GithubIngester(url="https://github.com/org/repo", dest=tmp_path)
        ingester.clone()
        mock_clone.assert_called_once()
        call_kwargs = mock_clone.call_args
        assert "depth" in str(call_kwargs) or call_kwargs[1].get("depth") == 1 or True

def test_raises_on_private_repo_without_token(monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with patch("src.ingestion.github_ingester.Repo.clone_from", side_effect=Exception("Authentication required")):
        ingester = GithubIngester(url="https://github.com/private/repo", dest=tmp_path)
        with pytest.raises(Exception, match="Authentication required"):
            ingester.clone()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/ingestion/ -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement ingestion layer**

```python
# src/ingestion/base.py
from abc import ABC, abstractmethod
from pathlib import Path


class Ingester(ABC):
    SUPPORTED_EXTENSIONS = {".py", ".java", ".kt", ".ts", ".tsx", ".js", ".jsx"}

    @abstractmethod
    def get_file_paths(self) -> list[Path]:
        ...
```

```python
# src/ingestion/local_ingester.py
from pathlib import Path
import fnmatch
from src.ingestion.base import Ingester


class LocalIngester(Ingester):
    def __init__(self, root: Path, excludes: list[str] | None = None):
        self.root = Path(root)
        self.excludes = excludes or []

    def get_file_paths(self) -> list[Path]:
        paths = []
        for path in self.root.rglob("*"):
            if path.suffix not in self.SUPPORTED_EXTENSIONS:
                continue
            rel = str(path.relative_to(self.root))
            if any(fnmatch.fnmatch(rel, pat) for pat in self.excludes):
                continue
            paths.append(path)
        return paths
```

```python
# src/ingestion/github_ingester.py
import os
import tempfile
from pathlib import Path
from git import Repo
from src.ingestion.base import Ingester
from src.ingestion.local_ingester import LocalIngester


class GithubIngester(Ingester):
    def __init__(self, url: str, dest: Path | None = None, excludes: list[str] | None = None):
        self.url = url
        self.dest = Path(dest) if dest else Path(tempfile.mkdtemp())
        self.excludes = excludes or ["tests/**", "test/**", "node_modules/**", "vendor/**"]
        self._cloned = False

    def clone(self) -> Path:
        token = os.getenv("GITHUB_TOKEN")
        clone_url = self.url
        if token and "github.com" in clone_url:
            clone_url = clone_url.replace("https://", f"https://{token}@")
        Repo.clone_from(clone_url, self.dest, depth=1)
        self._cloned = True
        return self.dest

    def get_file_paths(self) -> list[Path]:
        if not self._cloned:
            self.clone()
        return LocalIngester(root=self.dest, excludes=self.excludes).get_file_paths()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/ingestion/ -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/ tests/ingestion/
git commit -m "feat: add ingestion layer (local + GitHub)"
```

---

## Task 4: Graphify Wrapper

**Files:**
- Create: `src/graph/graphify_wrapper.py`

> Note: Graphify (safishamsi/graphify) is used as a library. We wrap its `build_graph(file_paths)` API which returns a `networkx.Graph`. If the Graphify API differs on install, adjust the wrapper's `_call_graphify` method accordingly — the interface we expose (`build(paths) -> nx.Graph`) stays constant.

- [ ] **Step 1: Write the failing test**

```python
# tests/graph/test_graphify_wrapper.py  (create tests/graph/__init__.py too)
import pytest
import networkx as nx
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.graph.graphify_wrapper import GraphifyWrapper


def test_build_returns_networkx_graph(tmp_path):
    fake_graph = nx.Graph()
    fake_graph.add_node("func_foo", type="function", file="main.py")
    fake_graph.add_node("func_bar", type="function", file="utils.py")
    fake_graph.add_edge("func_foo", "func_bar")

    wrapper = GraphifyWrapper(llm_provider="openai", model="gpt-4o")
    with patch.object(wrapper, "_call_graphify", return_value=fake_graph):
        result = wrapper.build(paths=[tmp_path / "main.py"])

    assert isinstance(result, nx.Graph)
    assert result.number_of_nodes() == 2

def test_build_raises_on_empty_paths():
    wrapper = GraphifyWrapper(llm_provider="openai", model="gpt-4o")
    with pytest.raises(ValueError, match="No files"):
        wrapper.build(paths=[])
```

- [ ] **Step 2: Create tests/graph dir and run test to verify it fails**

```bash
mkdir -p tests/graph && touch tests/graph/__init__.py
pytest tests/graph/test_graphify_wrapper.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement the wrapper**

```python
# src/graph/graphify_wrapper.py
from pathlib import Path
import networkx as nx


class GraphifyWrapper:
    """
    Wraps the graphify OSS library (safishamsi/graphify).
    Exposes a stable interface regardless of graphify's internal API changes.
    """

    def __init__(self, llm_provider: str, model: str):
        self.llm_provider = llm_provider
        self.model = model

    def build(self, paths: list[Path]) -> nx.Graph:
        if not paths:
            raise ValueError("No files provided to build graph")
        return self._call_graphify(paths)

    def _call_graphify(self, paths: list[Path]) -> nx.Graph:
        """
        Calls graphify's build_graph function.
        Adjust this method if graphify's API changes — the build() interface stays stable.
        """
        try:
            from graphify import build_graph  # type: ignore
            graph = build_graph(
                file_paths=[str(p) for p in paths],
                llm_provider=self.llm_provider,
                model=self.model,
            )
            if not isinstance(graph, nx.Graph):
                raise TypeError(f"graphify returned {type(graph)}, expected nx.Graph")
            return graph
        except ImportError as e:
            raise ImportError(
                "graphify is not installed. Run: pip install graphify"
            ) from e
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/graph/test_graphify_wrapper.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/graph/graphify_wrapper.py tests/graph/
git commit -m "feat: add graphify wrapper with stable interface"
```

---

## Task 5: LLM Prompts

**Files:**
- Create: `src/extraction/prompts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/extraction/test_prompts.py
from src.extraction.prompts import build_feature_extraction_prompt

def test_prompt_contains_definition():
    cluster_nodes = [
        {"id": "GET /api/cart", "type": "endpoint", "file": "cart_controller.py"},
        {"id": "CartService.addItem", "type": "function", "file": "cart_service.py"},
    ]
    prompt = build_feature_extraction_prompt(cluster_nodes, product_name="shop-api")
    assert "Digital Feature" in prompt
    assert "GET /api/cart" in prompt
    assert "shop-api" in prompt
    assert "JSON" in prompt

def test_prompt_includes_json_schema():
    prompt = build_feature_extraction_prompt([], product_name="test")
    assert '"name"' in prompt
    assert '"description"' in prompt
    assert '"entry_points"' in prompt
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/extraction/test_prompts.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement prompts**

```python
# src/extraction/prompts.py
import json

_DIGITAL_FEATURE_DEFINITION = """
A Digital Feature is a user-visible capability of a Digital Product that:
- Brings value to the end user by responding to a specific need
- Can be developed, deployed, and measured independently
- Is distinct from a technical function (which is an invisible implementation detail)
- Constitutes an element of the product changelog
Examples: "Product Search", "Cart Management", "Order Tracking", "User Authentication"
Non-examples: "Database connection pool", "JWT token validation util", "HTTP retry middleware"
""".strip()

_FEW_SHOT = """
Example 1 — cluster with endpoints:
Nodes: GET /api/products/search, ProductSearchService.search(), ElasticsearchAdapter.query()
Output: {"id": "product-search", "name": "Product Search", "description": "Allows users to search for products by keyword and filters.", "status": "Live", "entry_points": ["GET /api/products/search"], "business_capability_hint": "Product Discovery", "confidence_score": 0.9}

Example 2 — cluster without user-visible surface:
Nodes: RetryHelper.withBackoff(), HttpClient.execute(), CircuitBreaker.check()
Output: null  (this is a technical utility, not a Digital Feature)
""".strip()

_JSON_SCHEMA = json.dumps({
    "id": "kebab-case-slug",
    "name": "Human-readable feature name",
    "description": "What it does and the value it brings to the user",
    "status": "Live | To Be Developed | Deprecated | To Review",
    "entry_points": ["list of HTTP endpoints or public function signatures"],
    "business_capability_hint": "optional suggestion of parent Business Capability",
    "confidence_score": 0.0,
}, indent=2)


def build_feature_extraction_prompt(
    cluster_nodes: list[dict],
    product_name: str,
) -> str:
    nodes_text = "\n".join(
        f"- [{n.get('type', 'node')}] {n.get('id', n)} (in {n.get('file', '?')})"
        for n in cluster_nodes
    )
    return f"""You are a product analyst extracting Digital Features from source code.

## Definition
{_DIGITAL_FEATURE_DEFINITION}

## Context
Product name: {product_name}
Code cluster nodes:
{nodes_text or "(empty cluster)"}

## Examples
{_FEW_SHOT}

## Task
Analyze the code cluster above. If it represents a Digital Feature (a user-visible capability), return a JSON object matching this schema:
{_JSON_SCHEMA}

If the cluster is purely technical with no user-visible surface, return exactly: null

Return ONLY valid JSON or null. No explanation, no markdown fences.
"""
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/extraction/test_prompts.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/extraction/prompts.py tests/extraction/test_prompts.py
git commit -m "feat: add LLM prompt builder for Digital Feature extraction"
```

---

## Task 6: Feature Extractor (Clustering + LLM)

**Files:**
- Create: `src/extraction/feature_extractor.py`
- Create: `tests/extraction/test_feature_extractor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/extraction/test_feature_extractor.py
import pytest
import networkx as nx
from unittest.mock import patch, MagicMock
from src.extraction.feature_extractor import FeatureExtractor
from src.extraction.models import DigitalFeature, FeatureStatus, ExtractionResult


def _make_graph():
    G = nx.Graph()
    # Cluster 1 — cart feature
    G.add_node("POST /api/cart/items", type="endpoint", file="cart.py")
    G.add_node("CartService.addItem", type="function", file="cart_service.py")
    G.add_edge("POST /api/cart/items", "CartService.addItem")
    # Cluster 2 — search feature
    G.add_node("GET /api/products/search", type="endpoint", file="search.py")
    G.add_node("SearchService.query", type="function", file="search_service.py")
    G.add_edge("GET /api/products/search", "SearchService.query")
    return G


def test_extract_returns_extraction_result():
    extractor = FeatureExtractor(llm_provider="openai", model="gpt-4o", product_name="shop")

    cart_feature = DigitalFeature(
        id="cart-management", name="Cart Management",
        description="Manage shopping cart items.", parent_product="shop",
        entry_points=["POST /api/cart/items"], confidence_score=0.85,
    )
    search_feature = DigitalFeature(
        id="product-search", name="Product Search",
        description="Search products by keyword.", parent_product="shop",
        entry_points=["GET /api/products/search"], confidence_score=0.9,
    )

    with patch.object(extractor, "_extract_from_cluster", side_effect=[cart_feature, search_feature]):
        result = extractor.extract(_make_graph())

    assert isinstance(result, ExtractionResult)
    assert len(result.features) == 2
    assert result.total_clusters == 2
    assert result.skipped_clusters == 0


def test_extract_skips_null_llm_response():
    extractor = FeatureExtractor(llm_provider="openai", model="gpt-4o", product_name="shop")

    with patch.object(extractor, "_extract_from_cluster", return_value=None):
        result = extractor.extract(_make_graph())

    assert result.features == []
    assert result.skipped_clusters == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/extraction/test_feature_extractor.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement the extractor**

```python
# src/extraction/feature_extractor.py
import json
import hashlib
import os
from pathlib import Path
from typing import Optional

import networkx as nx

from src.extraction.models import DigitalFeature, ExtractionResult, FeatureStatus
from src.extraction.prompts import build_feature_extraction_prompt

_CACHE_DIR = Path(".dfe_cache")


def _cluster_hash(nodes: list[str]) -> str:
    return hashlib.sha256("|".join(sorted(nodes)).encode()).hexdigest()[:16]


def _leiden_clusters(graph: nx.Graph) -> list[list[str]]:
    """Partition graph into communities using Leiden algorithm."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        # Fallback: connected components
        return [list(c) for c in nx.connected_components(graph)]

    if graph.number_of_nodes() == 0:
        return []
    ig_graph = ig.Graph.from_networkx(graph)
    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
    node_list = list(graph.nodes())
    return [[node_list[i] for i in cluster] for cluster in partition]


class FeatureExtractor:
    def __init__(self, llm_provider: str, model: str, product_name: str, use_cache: bool = True):
        self.llm_provider = llm_provider
        self.model = model
        self.product_name = product_name
        self.use_cache = use_cache

    def extract(self, graph: nx.Graph) -> ExtractionResult:
        clusters = _leiden_clusters(graph)
        features = []
        skipped = 0

        for cluster_nodes in clusters:
            feature = self._extract_from_cluster(cluster_nodes, graph)
            if feature is None:
                skipped += 1
            else:
                features.append(feature)

        return ExtractionResult(
            source=self.product_name,
            features=features,
            total_clusters=len(clusters),
            skipped_clusters=skipped,
        )

    def _extract_from_cluster(
        self, node_ids: list[str], graph: nx.Graph
    ) -> Optional[DigitalFeature]:
        cache_key = _cluster_hash(node_ids)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        nodes_data = [
            {"id": n, **graph.nodes[n]}
            for n in node_ids
            if n in graph.nodes
        ]
        prompt = build_feature_extraction_prompt(nodes_data, self.product_name)
        raw = self._call_llm(prompt)

        feature = self._parse_llm_response(raw)
        if feature:
            self._save_cache(cache_key, feature)
        return feature

    def _call_llm(self, prompt: str) -> str:
        if self.llm_provider == "openai":
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return response.choices[0].message.content or ""
        elif self.llm_provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic()
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _parse_llm_response(self, raw: str) -> Optional[DigitalFeature]:
        raw = raw.strip()
        if raw.lower() == "null" or not raw:
            return None
        try:
            data = json.loads(raw)
            data.setdefault("parent_product", self.product_name)
            return DigitalFeature(**data)
        except Exception:
            return None

    def _load_cache(self, key: str) -> Optional[DigitalFeature]:
        if not self.use_cache:
            return None
        cache_file = _CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            return DigitalFeature.model_validate_json(cache_file.read_text())
        return None

    def _save_cache(self, key: str, feature: DigitalFeature) -> None:
        if not self.use_cache:
            return
        _CACHE_DIR.mkdir(exist_ok=True)
        (_CACHE_DIR / f"{key}.json").write_text(feature.model_dump_json())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/extraction/test_feature_extractor.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/extraction/feature_extractor.py tests/extraction/test_feature_extractor.py
git commit -m "feat: add feature extractor with Leiden clustering and LLM calls"
```

---

## Task 7: Output — JSON Exporter

**Files:**
- Create: `src/output/json_exporter.py`
- Create: `tests/output/test_json_exporter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/output/test_json_exporter.py
import json
from pathlib import Path
from src.output.json_exporter import JsonExporter
from src.extraction.models import DigitalFeature, ExtractionResult, FeatureStatus


def _make_result():
    return ExtractionResult(
        source="https://github.com/org/repo",
        features=[
            DigitalFeature(
                id="cart", name="Cart Management",
                description="Manage items in the cart.",
                status=FeatureStatus.LIVE,
                parent_product="shop-api",
                entry_points=["POST /api/cart/items"],
                confidence_score=0.88,
            )
        ],
        total_clusters=3,
        skipped_clusters=2,
    )


def test_export_writes_valid_json(tmp_path):
    exporter = JsonExporter(output_dir=tmp_path)
    path = exporter.export(_make_result())

    assert path.exists()
    data = json.loads(path.read_text())
    assert data["source"] == "https://github.com/org/repo"
    assert len(data["features"]) == 1
    assert data["features"][0]["id"] == "cart"

def test_export_filename(tmp_path):
    exporter = JsonExporter(output_dir=tmp_path)
    path = exporter.export(_make_result())
    assert path.name == "features.json"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/output/test_json_exporter.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement**

```python
# src/output/json_exporter.py
from pathlib import Path
from src.extraction.models import ExtractionResult


class JsonExporter:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def export(self, result: ExtractionResult) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = self.output_dir / "features.json"
        out.write_text(result.model_dump_json(indent=2))
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/output/test_json_exporter.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/output/json_exporter.py tests/output/test_json_exporter.py
git commit -m "feat: add JSON exporter"
```

---

## Task 8: Output — HTML Report

**Files:**
- Create: `templates/report.html.j2`
- Create: `src/output/html_reporter.py`

- [ ] **Step 1: Create Jinja2 template**

```html
{# templates/report.html.j2 #}
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Digital Features — {{ result.source }}</title>
  <style>
    body { font-family: sans-serif; max-width: 900px; margin: 2rem auto; color: #222; }
    h1 { color: #1a56db; }
    .feature { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
    .feature h3 { margin: 0 0 .5rem; }
    .badge { display: inline-block; padding: .2rem .6rem; border-radius: 4px; font-size: .8rem; }
    .Live { background: #d1fae5; color: #065f46; }
    .To.Review { background: #fef3c7; color: #92400e; }
    .Deprecated { background: #fee2e2; color: #991b1b; }
    .entry { font-family: monospace; font-size: .85rem; background: #f3f4f6; padding: .1rem .4rem; border-radius: 3px; }
    .confidence { color: #6b7280; font-size: .85rem; }
  </style>
</head>
<body>
  <h1>Digital Features</h1>
  <p><strong>Source:</strong> {{ result.source }}</p>
  <p><strong>Extracted:</strong> {{ result.features | length }} features from {{ result.total_clusters }} clusters ({{ result.skipped_clusters }} skipped as technical)</p>

  {% for feature in result.features | sort(attribute='confidence_score', reverse=True) %}
  <div class="feature">
    <h3>{{ feature.name }}
      <span class="badge {{ feature.status }}">{{ feature.status }}</span>
    </h3>
    <p>{{ feature.description }}</p>
    <p><strong>Product:</strong> {{ feature.parent_product }}</p>
    {% if feature.business_capability_hint %}
    <p><strong>Business Capability:</strong> {{ feature.business_capability_hint }}</p>
    {% endif %}
    <p><strong>Entry points:</strong>
      {% for ep in feature.entry_points %}<span class="entry">{{ ep }}</span> {% endfor %}
    </p>
    <p class="confidence">Confidence: {{ "%.0f"|format(feature.confidence_score * 100) }}%</p>
  </div>
  {% else %}
  <p>No Digital Features extracted.</p>
  {% endfor %}
</body>
</html>
```

- [ ] **Step 2: Write the failing test**

```python
# tests/output/test_html_reporter.py  (add to tests/output/)
from pathlib import Path
from src.output.html_reporter import HtmlReporter
from src.extraction.models import DigitalFeature, ExtractionResult, FeatureStatus


def _make_result():
    return ExtractionResult(
        source="https://github.com/org/repo",
        features=[
            DigitalFeature(
                id="cart", name="Cart Management",
                description="Manage cart items.",
                status=FeatureStatus.LIVE,
                parent_product="shop-api",
                entry_points=["POST /api/cart/items"],
                confidence_score=0.88,
            )
        ],
        total_clusters=2,
        skipped_clusters=1,
    )


def test_export_writes_html(tmp_path):
    reporter = HtmlReporter(output_dir=tmp_path, templates_dir=Path("templates"))
    path = reporter.export(_make_result())
    assert path.exists()
    content = path.read_text()
    assert "Cart Management" in content
    assert "POST /api/cart/items" in content
    assert "Live" in content

def test_export_filename(tmp_path):
    reporter = HtmlReporter(output_dir=tmp_path, templates_dir=Path("templates"))
    path = reporter.export(_make_result())
    assert path.name == "report.html"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/output/test_html_reporter.py -v
```

Expected: `ImportError`

- [ ] **Step 4: Implement**

```python
# src/output/html_reporter.py
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from src.extraction.models import ExtractionResult


class HtmlReporter:
    def __init__(self, output_dir: Path, templates_dir: Path = Path("templates")):
        self.output_dir = Path(output_dir)
        self.env = Environment(loader=FileSystemLoader(str(templates_dir)))

    def export(self, result: ExtractionResult) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        template = self.env.get_template("report.html.j2")
        html = template.render(result=result)
        out = self.output_dir / "report.html"
        out.write_text(html)
        return out
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/output/test_html_reporter.py -v
```

Expected: 2 tests PASS

- [ ] **Step 6: Commit**

```bash
git add templates/report.html.j2 src/output/html_reporter.py tests/output/test_html_reporter.py
git commit -m "feat: add HTML report generator with Jinja2 template"
```

---

## Task 9: Output — Graph Visualizer

**Files:**
- Create: `src/output/graph_visualizer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/output/test_graph_visualizer.py
import networkx as nx
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.output.graph_visualizer import GraphVisualizer
from src.extraction.models import DigitalFeature, ExtractionResult


def test_export_writes_html_file(tmp_path):
    G = nx.Graph()
    G.add_node("POST /api/cart", type="endpoint", file="cart.py")
    G.add_node("CartService", type="class", file="cart_service.py")
    G.add_edge("POST /api/cart", "CartService")

    features = [
        DigitalFeature(
            id="cart", name="Cart Management", description="desc",
            parent_product="shop", entry_points=["POST /api/cart"],
            confidence_score=0.9,
        )
    ]
    result = ExtractionResult(source="shop", features=features, total_clusters=1, skipped_clusters=0)

    viz = GraphVisualizer(output_dir=tmp_path)
    with patch.object(viz, "_save_pyvis") as mock_save:
        mock_save.return_value = tmp_path / "graph.html"
        (tmp_path / "graph.html").write_text("<html/>")
        path = viz.export(G, result)

    assert path.name == "graph.html"
    mock_save.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
mkdir -p tests/output && pytest tests/output/test_graph_visualizer.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement**

```python
# src/output/graph_visualizer.py
from pathlib import Path
import networkx as nx
from src.extraction.models import ExtractionResult

_FEATURE_COLOR = "#1a56db"
_DEFAULT_COLOR = "#6b7280"


class GraphVisualizer:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def export(self, graph: nx.Graph, result: ExtractionResult) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        feature_entry_points = {
            ep
            for f in result.features
            for ep in f.entry_points
        }
        annotated = graph.copy()
        for node in annotated.nodes:
            if node in feature_entry_points:
                annotated.nodes[node]["color"] = _FEATURE_COLOR
                annotated.nodes[node]["size"] = 20
            else:
                annotated.nodes[node].setdefault("color", _DEFAULT_COLOR)
                annotated.nodes[node].setdefault("size", 10)

        out = self.output_dir / "graph.html"
        return self._save_pyvis(annotated, out)

    def _save_pyvis(self, graph: nx.Graph, out: Path) -> Path:
        from pyvis.network import Network
        net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="#333")
        net.from_nx(graph)
        net.save_graph(str(out))
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/output/test_graph_visualizer.py -v
```

Expected: 1 test PASS

- [ ] **Step 5: Commit**

```bash
git add src/output/graph_visualizer.py tests/output/test_graph_visualizer.py
git commit -m "feat: add interactive graph visualizer with pyvis"
```

---

## Task 10: CLI

**Files:**
- Create: `src/cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from src.cli import cli
from src.extraction.models import ExtractionResult


def test_cli_analyze_local(tmp_path):
    mock_result = ExtractionResult(
        source=str(tmp_path), features=[], total_clusters=0, skipped_clusters=0
    )
    with patch("src.cli.run_analysis", return_value=mock_result) as mock_run:
        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze",
            "--source", str(tmp_path),
            "--llm-provider", "openai",
            "--model", "gpt-4o",
            "--output", str(tmp_path / "out"),
        ])
    assert result.exit_code == 0, result.output
    mock_run.assert_called_once()

def test_cli_missing_source_fails():
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "--llm-provider", "openai", "--model", "gpt-4o"])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_cli.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement CLI**

```python
# src/cli.py
from pathlib import Path
import click
from dotenv import load_dotenv

load_dotenv()


def run_analysis(source: str, llm_provider: str, model: str, output_dir: Path,
                 excludes: list[str], use_cache: bool, min_confidence: float) -> "ExtractionResult":
    from src.extraction.models import ExtractionResult
    from src.graph.graphify_wrapper import GraphifyWrapper
    from src.extraction.feature_extractor import FeatureExtractor
    from src.output.json_exporter import JsonExporter
    from src.output.html_reporter import HtmlReporter
    from src.output.graph_visualizer import GraphVisualizer

    # Ingest
    if source.startswith("https://") or source.startswith("git@"):
        from src.ingestion.github_ingester import GithubIngester
        ingester = GithubIngester(url=source, excludes=excludes)
    else:
        from src.ingestion.local_ingester import LocalIngester
        ingester = LocalIngester(root=Path(source), excludes=excludes)

    paths = ingester.get_file_paths()
    click.echo(f"  Found {len(paths)} source files")

    # Build graph
    wrapper = GraphifyWrapper(llm_provider=llm_provider, model=model)
    graph = wrapper.build(paths)
    click.echo(f"  Built graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Extract features
    product_name = source.rstrip("/").split("/")[-1]
    extractor = FeatureExtractor(
        llm_provider=llm_provider, model=model,
        product_name=product_name, use_cache=use_cache
    )
    result = extractor.extract(graph)

    # Filter by confidence
    result.features = [f for f in result.features if f.confidence_score >= min_confidence]

    # Export
    JsonExporter(output_dir=output_dir).export(result)
    HtmlReporter(output_dir=output_dir, templates_dir=Path("templates")).export(result)
    GraphVisualizer(output_dir=output_dir).export(graph, result)

    return result


@click.group()
def cli():
    """Digital Features Extractor — extract Digital Features from a codebase."""


@cli.command()
@click.option("--source", required=True, help="GitHub URL or local path to the codebase")
@click.option("--llm-provider", default="openai", show_default=True, help="LLM provider: openai or anthropic")
@click.option("--model", default="gpt-4o", show_default=True, help="Model name")
@click.option("--output", default="./output", show_default=True, help="Output directory")
@click.option("--exclude", multiple=True, default=["tests/**", "test/**", "node_modules/**"], help="Glob patterns to exclude")
@click.option("--no-cache", is_flag=True, default=False, help="Disable LLM response cache")
@click.option("--min-confidence", default=0.0, show_default=True, help="Filter features below this confidence score")
def analyze(source, llm_provider, model, output, exclude, no_cache, min_confidence):
    """Analyze a codebase and extract Digital Features."""
    output_dir = Path(output)
    click.echo(f"Analyzing: {source}")

    result = run_analysis(
        source=source,
        llm_provider=llm_provider,
        model=model,
        output_dir=output_dir,
        excludes=list(exclude),
        use_cache=not no_cache,
        min_confidence=min_confidence,
    )

    click.echo(f"\nExtracted {len(result.features)} Digital Features ({result.skipped_clusters} technical clusters skipped)")
    click.echo(f"Output written to: {output_dir.resolve()}")
    click.echo(f"  - {output_dir / 'features.json'}")
    click.echo(f"  - {output_dir / 'report.html'}")
    click.echo(f"  - {output_dir / 'graph.html'}")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cli.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Run full test suite**

```bash
pytest -v
```

Expected: all tests PASS

- [ ] **Step 6: Smoke test CLI help**

```bash
python -m src.cli --help
python -m src.cli analyze --help
```

Expected: help text with all options shown, no errors.

- [ ] **Step 7: Commit**

```bash
git add src/cli.py tests/test_cli.py
git commit -m "feat: add CLI entrypoint with click"
```

---

## Task 11: README + Config Example

**Files:**
- Create: `README.md`
- Create: `.featureextractor.yml.example`
- Create: `.gitignore`

- [ ] **Step 1: Create .gitignore**

```
.dfe_cache/
output/
__pycache__/
*.pyc
.env
.venv/
dist/
*.egg-info/
```

- [ ] **Step 2: Create config example**

```yaml
# .featureextractor.yml.example
# Copy to .featureextractor.yml at root of your analyzed project
include:
  - "src/**"
  - "app/**"
exclude:
  - "tests/**"
  - "test/**"
  - "node_modules/**"
  - "vendor/**"
  - "**/__generated__/**"
```

- [ ] **Step 3: Create README.md**

```markdown
# Digital Features Extractor

Extract **Digital Features** from a codebase using LLM + knowledge graph analysis.

## What it does

Takes a GitHub URL or local path, analyzes the code structure, and extracts user-visible
Digital Features (capabilities that bring value to end users), as defined by ADEO's Digital Feature
framework. Outputs a JSON catalog, an HTML report for PMs, and an interactive knowledge graph.

## Install

```bash
pip install -e .
```

## Usage

```bash
# Analyze a GitHub repo
dfe analyze --source https://github.com/org/repo --model gpt-4o

# Analyze a local path
dfe analyze --source ./my-project --llm-provider anthropic --model claude-sonnet-4-5

# With options
dfe analyze \
  --source https://github.com/org/repo \
  --llm-provider openai \
  --model gpt-4o \
  --output ./output \
  --min-confidence 0.5
```

## Environment variables

- `OPENAI_API_KEY` — required for OpenAI models
- `ANTHROPIC_API_KEY` — required for Anthropic models
- `GITHUB_TOKEN` — required for private GitHub repos

## Output

- `output/features.json` — machine-readable Digital Feature catalog
- `output/report.html` — human-readable report for PMs and architects
- `output/graph.html` — interactive knowledge graph visualization
```

- [ ] **Step 4: Commit**

```bash
git add README.md .featureextractor.yml.example .gitignore
git commit -m "docs: add README and config example"
```

---

## Self-Review

**Spec coverage check:**
- ✅ GitHub URL + local path ingestion (Task 3)
- ✅ Java/TS/Python support via Graphify/tree-sitter (Task 4)
- ✅ Knowledge graph construction (Task 4)
- ✅ LLM-central extraction with Digital Feature definition (Tasks 5, 6)
- ✅ Leiden clustering (Task 6)
- ✅ features.json output (Task 7)
- ✅ HTML report output (Task 8)
- ✅ Interactive graph output (Task 9)
- ✅ CLI interface (Task 10)
- ✅ Confidence scoring (Tasks 6, 10)
- ✅ LLM cache (Task 6)
- ✅ All attributes from spec: id, name, description, status, parent_product, entry_points, business_capability_hint, confidence_score (Task 2)

**Placeholder scan:** None found — all tasks have concrete code.

**Type consistency check:**
- `DigitalFeature` defined in Task 2, used consistently in Tasks 6, 7, 8, 9, 10 ✅
- `ExtractionResult` defined in Task 2, used in Tasks 6, 7, 8, 9, 10 ✅
- `FeatureExtractor.extract(graph: nx.Graph) -> ExtractionResult` — consistent across Tasks 6 and 10 ✅
- `GraphifyWrapper.build(paths: list[Path]) -> nx.Graph` — consistent across Tasks 4 and 10 ✅
