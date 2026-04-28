from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import networkx as nx

try:
    from graphify import extract as _gfy_extract, build_from_json as _gfy_build, cluster as _gfy_cluster  # type: ignore

    _GRAPHIFY_AVAILABLE = True
except Exception:
    _GRAPHIFY_AVAILABLE = False


def _fallback_extract(files: list[Path]) -> dict:
    nodes: list[dict] = []
    edges: list[dict] = []
    file_ids: dict[Path, str] = {}

    def _file_id(p: Path) -> str:
        return p.stem.replace("-", "_").lower()

    for f in files:
        fid = _file_id(f)
        file_ids[f] = fid
        nodes.append({"id": fid, "type": "file", "name": f.name, "source_file": str(f)})

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

    return {"nodes": nodes, "edges": edges}


def _fallback_build(extraction: dict) -> nx.Graph:
    G: nx.Graph = nx.DiGraph()
    for n in extraction.get("nodes", []):
        G.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
    for e in extraction.get("edges", []):
        src, tgt = e.get("source"), e.get("target")
        if src and tgt and src in G and tgt in G:
            G.add_edge(src, tgt, **{k: v for k, v in e.items() if k not in ("source", "target")})
    return G


class GraphifyWrapper:
    """Builds and analyses a NetworkX graph from source files."""

    def __init__(self, llm_client: Any | None = None) -> None:
        self._llm_client = llm_client

    def build_graph(self, files: list[Path]) -> nx.Graph:
        """Build a NetworkX graph from source files.

        Nodes: files, classes, functions — each with ``type``, ``name``, ``source_file`` attributes.
        Edges: import/call relationships between nodes.
        """
        if _GRAPHIFY_AVAILABLE:
            try:
                extraction = _gfy_extract(files)
                return _gfy_build(extraction)
            except Exception:
                pass
        extraction = _fallback_extract(files)
        return _fallback_build(extraction)

    def get_clusters(self, graph: nx.Graph) -> dict[str, list[str]]:
        """Return ``{cluster_id: [node_id, ...]}`` via community detection."""
        if graph.number_of_nodes() == 0:
            return {}

        undirected = graph.to_undirected() if graph.is_directed() else graph

        if _GRAPHIFY_AVAILABLE:
            try:
                raw: dict[int, list[str]] = _gfy_cluster(undirected)
                return {str(k): v for k, v in raw.items()}
            except Exception:
                pass

        components = list(nx.connected_components(undirected))
        return {str(i): list(nodes) for i, nodes in enumerate(components)}

    def to_dict(self, graph: nx.Graph) -> dict:
        """Return a JSON-serialisable ``{"nodes": [...], "edges": [...]}`` dict."""
        def _safe(v: Any) -> Any:
            return v if isinstance(v, (str, int, float, bool, type(None))) else str(v)

        nodes = [
            {"id": n, **{k: _safe(val) for k, val in attrs.items()}}
            for n, attrs in graph.nodes(data=True)
        ]
        edges = [
            {"source": u, "target": v, **{k: _safe(val) for k, val in data.items()}}
            for u, v, data in graph.edges(data=True)
        ]
        result = {"nodes": nodes, "edges": edges}
        json.dumps(result)
        return result
