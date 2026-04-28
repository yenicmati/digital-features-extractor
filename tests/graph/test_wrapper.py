import json
from pathlib import Path

import pytest

from src.graph import GraphifyWrapper


@pytest.fixture()
def two_linked_files(tmp_path: Path) -> tuple[Path, Path]:
    alpha = tmp_path / "alpha.py"
    beta = tmp_path / "beta.py"
    alpha.write_text("import beta\n\ndef hello(): pass\n")
    beta.write_text("def world(): pass\n")
    return alpha, beta


def test_build_graph_nodes_include_both_files(two_linked_files):
    alpha, beta = two_linked_files
    wrapper = GraphifyWrapper()
    graph = wrapper.build_graph([alpha, beta])

    node_names = {data.get("name") for _, data in graph.nodes(data=True)}
    assert "alpha.py" in node_names
    assert "beta.py" in node_names


def test_build_graph_edge_between_files(two_linked_files):
    alpha, beta = two_linked_files
    wrapper = GraphifyWrapper()
    graph = wrapper.build_graph([alpha, beta])

    assert graph.number_of_nodes() >= 2
    assert graph.number_of_edges() >= 1


def test_get_clusters_returns_non_empty_dict(two_linked_files):
    alpha, beta = two_linked_files
    wrapper = GraphifyWrapper()
    graph = wrapper.build_graph([alpha, beta])
    clusters = wrapper.get_clusters(graph)

    assert isinstance(clusters, dict)
    assert len(clusters) >= 1
    for key, members in clusters.items():
        assert isinstance(key, str)
        assert isinstance(members, list)


def test_get_clusters_empty_graph_returns_empty():
    import networkx as nx
    wrapper = GraphifyWrapper()
    assert wrapper.get_clusters(nx.Graph()) == {}


def test_to_dict_has_nodes_and_edges_keys(two_linked_files):
    alpha, beta = two_linked_files
    wrapper = GraphifyWrapper()
    graph = wrapper.build_graph([alpha, beta])
    result = wrapper.to_dict(graph)

    assert "nodes" in result
    assert "edges" in result
    assert isinstance(result["nodes"], list)
    assert isinstance(result["edges"], list)


def test_to_dict_is_json_serialisable(two_linked_files):
    alpha, beta = two_linked_files
    wrapper = GraphifyWrapper()
    graph = wrapper.build_graph([alpha, beta])
    result = wrapper.to_dict(graph)

    serialised = json.dumps(result)
    assert len(serialised) > 0
