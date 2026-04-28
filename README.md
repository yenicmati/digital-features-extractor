# Digital Features Extractor (DFE)

Digital Features Extractor is an LLM-powered tool designed to automatically identify and extract high-level **Digital Features** from a codebase. By analyzing code structures and relationships through a knowledge graph, DFE bridges the gap between technical implementation and business value.

## What is a Digital Feature?

A **Digital Feature** is a user-visible capability of a digital product that delivers specific business value. Unlike technical functions or components (e.g., "Database Schema" or "API Endpoint"), a Digital Feature describes what a user can actually *do* (e.g., "One-Click Checkout" or "User Profile Management").

## Installation

DFE requires Python 3.11 or higher.

1. **Clone the repository** (or navigate to your local copy).
2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install the package**:
   ```bash
   pip install -e .
   ```
4. **(Optional) Install graphify for richer graph analysis**:
   ```bash
   pip install git+https://github.com/safishamsi/graphify.git
   ```
   Without it, the tool falls back to a built-in AST-based graph builder.

## Quick Start

**With OpenAI:**
```bash
dfe analyze --source https://github.com/org/repo --api-key sk-...
```

**With GitHub Copilot subscription** (no extra cost):
```bash
dfe analyze --source https://github.com/org/repo \
  --api-key $(gh auth token) \
  --base-url https://api.githubcopilot.com \
  --model gpt-4o
```

**With a local directory:**
```bash
dfe analyze --source ./my-project --api-key $(gh auth token) \
  --base-url https://api.githubcopilot.com
```

Results are written to `./dfe-output/` by default.

## Usage Options

The `dfe analyze` command supports several options:

| Option | Short | Environment Variable | Description | Default |
| :--- | :--- | :--- | :--- | :--- |
| `--source` | `-s` | | GitHub URL or local directory path | (Required) |
| `--output-dir` | `-o` | | Directory where reports are saved | `./dfe-output` |
| `--provider` | | | LLM provider: `openai` or `anthropic` | `openai` |
| `--model` | | | LLM model name to use | `gpt-4o` |
| `--api-key` | | `OPENAI_API_KEY` | API key for the chosen provider | |
| `--base-url` | | `OPENAI_BASE_URL` | Custom API base URL (e.g. GitHub Copilot) | |
| `--cache-dir` | | | Cache directory for LLM responses | `None` |
| `--verbose` | `-v` | | Enable detailed console output | `False` |

### Using Anthropic

To use Claude models, specify the provider and model:

```bash
dfe analyze --source ./my-project --provider anthropic --model claude-3-5-sonnet-20241022 --api-key your-anthropic-key
```

## Outputs

After analysis, DFE generates three main files in the output directory:

1. **`features.json`**: A machine-readable list of extracted features, their descriptions, and associated code clusters.
2. **`report.html`**: A human-friendly documentation page summarizing the project's digital capabilities.
3. **`graph.html`**: An interactive visualization of the knowledge graph and how code components map to specific features.

## Development

To set up the development environment and run tests:

1. **Install dev dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
2. **Run tests**:
   ```bash
   pytest -v
   ```
