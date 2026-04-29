from __future__ import annotations

import sys
from pathlib import Path

import click

from src.extraction import FeatureExtractor, FeatureGrouper
from src.graph import GraphifyWrapper
from src.ingestion import GithubIngester, LocalIngester
from src.output import GraphVisualizer, HtmlReporter, JsonExporter


def _is_github_source(source: str) -> bool:
    return source.startswith("https://github.com") or source.startswith("git@")


def _build_llm_client(provider: str, api_key: str | None, base_url: str | None = None) -> object:
    if provider == "openai":
        try:
            import openai
        except ImportError:
            raise click.ClickException(
                "openai package is not installed. Run: pip install openai"
            )
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
            kwargs["default_headers"] = {"Copilot-Integration-Id": "vscode-chat"}
        return openai.OpenAI(**kwargs)
    elif provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise click.ClickException(
                "anthropic package is not installed. Run: pip install anthropic"
            )
        return anthropic.Anthropic(api_key=api_key)
    else:
        raise click.ClickException(f"Unknown provider: {provider}")


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--source", "-s", required=True, help="GitHub URL or local path")
@click.option("--output-dir", "-o", default="./dfe-output", help="Output directory")
@click.option("--model", default="gpt-4.1", help="LLM model to use")
@click.option("--api-key", envvar="OPENAI_API_KEY", help="OpenAI API key")
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic"]),
    default="openai",
    help="LLM provider",
)
@click.option("--cache-dir", default=None, help="Cache directory for LLM responses")
@click.option("--base-url", default=None, envvar="OPENAI_BASE_URL", help="Custom API base URL (e.g. https://api.githubcopilot.com)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def analyze(
    source: str,
    output_dir: str,
    model: str,
    api_key: str | None,
    provider: str,
    cache_dir: str | None,
    base_url: str | None,
    verbose: bool,
) -> None:
    out = Path(output_dir)
    cache_path: Path | None = Path(cache_dir) if cache_dir else None

    if _is_github_source(source):
        ingester = GithubIngester()
        if verbose:
            click.echo(f"Cloning repository: {source}")
    else:
        ingester = LocalIngester()
        if verbose:
            click.echo(f"Reading local path: {source}")

    try:
        files = ingester.ingest(source)
    except (ValueError, Exception) as exc:
        raise click.ClickException(str(exc)) from exc

    if verbose:
        click.echo(f"  → {len(files)} files ingested")

    llm_client = _build_llm_client(provider, api_key, base_url)

    if verbose:
        click.echo("Building knowledge graph…")
    wrapper = GraphifyWrapper()
    graph = wrapper.build_graph(files)
    clusters = wrapper.get_clusters(graph)

    if verbose:
        click.echo(f"  → {graph.number_of_nodes()} nodes, {len(clusters)} clusters")

    if verbose:
        click.echo("Extracting digital features via LLM…")
    extractor = FeatureExtractor(llm_client, model, cache_path)
    result = extractor.extract(clusters, graph, source=source)

    grouper = FeatureGrouper(llm_client, model)
    grouping = grouper.group(result)

    out.mkdir(parents=True, exist_ok=True)

    JsonExporter().export(result, out / "features.json", grouping=grouping)
    HtmlReporter().export(result, out / "report.html", source=source, grouping=grouping)
    GraphVisualizer().export(graph, result.features, out / "graph.html")

    if verbose:
        click.echo(f"  → Output written to {out}/")

    click.echo(
        f"✓ Found {len(result.features)} features in {result.total_clusters} clusters → {len(grouping.business_features)} business feature groups"
    )
