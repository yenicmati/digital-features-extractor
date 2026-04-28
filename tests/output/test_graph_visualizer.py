from pathlib import Path

import networkx as nx
import pytest

from src.extraction.models import DigitalFeature, FeatureStatus
from src.output.graph_visualizer import GraphVisualizer


@pytest.fixture
def sample_graph() -> nx.Graph:
    G = nx.DiGraph()
    G.add_node("file_a", type="file", name="file_a.py", source_file="/src/file_a.py")
    G.add_node("feature_1", type="feature", name="Feature One", source_file="/src/file_a.py")
    G.add_node("other_node", type="function", name="helper_function", source_file="/src/file_a.py")
    G.add_edge("file_a", "feature_1", type="contains")
    G.add_edge("file_a", "other_node", type="contains")
    return G


@pytest.fixture
def sample_features() -> list[DigitalFeature]:
    return [
        DigitalFeature(
            id="f1",
            name="Feature One",
            description="First feature description",
            status=FeatureStatus.LIVE,
            parent_product="Product A",
            entry_points=["/a"],
            confidence_score=0.8,
        ),
    ]


def test_output_file_created(tmp_path: Path, sample_graph: nx.Graph, sample_features: list[DigitalFeature]) -> None:
    visualizer = GraphVisualizer()
    out = tmp_path / "graph.html"
    visualizer.export(sample_graph, sample_features, out)
    assert out.exists()


def test_output_contains_html_tag(tmp_path: Path, sample_graph: nx.Graph, sample_features: list[DigitalFeature]) -> None:
    visualizer = GraphVisualizer()
    out = tmp_path / "graph.html"
    visualizer.export(sample_graph, sample_features, out)
    content = out.read_text(encoding="utf-8")
    assert "<html" in content.lower()


def test_output_contains_feature_description_when_node_name_matches(
    tmp_path: Path,
    sample_graph: nx.Graph,
    sample_features: list[DigitalFeature],
) -> None:
    visualizer = GraphVisualizer()
    out = tmp_path / "graph.html"
    visualizer.export(sample_graph, sample_features, out)
    content = out.read_text(encoding="utf-8")
    assert "First feature description" in content
