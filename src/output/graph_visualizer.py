from pathlib import Path

import networkx as nx
from pyvis.network import Network

from src.extraction.models import DigitalFeature


class GraphVisualizer:
    def __init__(self, height: str = "800px", width: str = "100%") -> None:
        self.height = height
        self.width = width

    def export(
        self,
        graph: nx.Graph,
        features: list[DigitalFeature],
        output_path: Path,
    ) -> None:
        feature_map: dict[str, DigitalFeature] = {f.name: f for f in features}
        net = Network(height=self.height, width=self.width, directed=graph.is_directed())

        feature_color = "#4ade80"
        default_color = "#9ca3af"

        for node_id, attrs in graph.nodes(data=True):
            node_name = str(attrs.get("name", node_id))
            matched = feature_map.get(node_name)
            color = feature_color if matched else default_color
            title = matched.description if matched else str(attrs.get("type", ""))
            net.add_node(node_id, label=node_name, color=color, title=title)

        for source, target, data in graph.edges(data=True):
            net.add_edge(source, target, **data)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        net.write_html(str(output_path))
