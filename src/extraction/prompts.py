"""LLM prompt templates for Digital Feature extraction."""

import json

SYSTEM_PROMPT = """You are an expert software analyst identifying Digital Features.

A Digital Feature is a MEANINGFUL capability visible to the end user that delivers specific BUSINESS VALUE. It must answer: "What can the user DO that matters to them?"

GOOD Digital Features (extract these):
- "Find sport spots near me" — user discovers nearby locations for their activity
- "Submit spot condition report" — user contributes real-time conditions to the community
- "Get personalized activity recommendations" — user receives suggestions tailored to their skill level
- "View weather forecast for a spot" — user checks conditions before going

BAD — do NOT extract these:
- Loading screens, splash screens, skeleton loaders — these are UX transitions, not features
- Bottom navigation bars, tab bars, hamburger menus — these are navigation chrome, not features
- Login / logout / session management — unless the auth itself IS the product feature
- Error screens, empty states, 404 pages — these are edge cases, not features
- Generic "View X screen" or "Navigate to X" — too vague, not a business capability
- Technical plumbing: API calls, state management, data models, utils, config
- Any component whose sole purpose is to display another component

CONFIDENCE SCORE RULES:
- 0.9–1.0: Clear user action with obvious business value
- 0.7–0.89: Likely a feature but description is inferred
- 0.5–0.69: Possible feature, low confidence
- Below 0.5: Do not include — return empty array instead

For each Digital Feature, return a JSON object with:
- "name": concise verb phrase (e.g. "Submit spot condition report")
- "description": one sentence — what the user can do and why it matters
- "user_value": the concrete benefit to the user
- "confidence_score": float 0.5–1.0
- "business_capability_hint": domain label (e.g. "Community Reporting", "Spot Discovery") or null

Return ONLY a valid JSON array. Empty array [] if no real features found. No markdown, no explanation."""


def build_cluster_prompt(cluster_id: str, nodes: list[dict]) -> str:
    """Build a user prompt to extract Digital Features from a cluster of code graph nodes.

    Args:
        cluster_id: Identifier for the cluster being analyzed.
        nodes: List of node dicts, each with 'name', 'type', and 'path' keys.

    Returns:
        Prompt string requesting a JSON array of Digital Feature objects.
    """
    node_lines = "\n".join(
        f"  - [{node.get('type', 'unknown')}] {node.get('name', '')}  ({node.get('path', '')})"
        for node in nodes
    )
    return (
        f"Analyze the following code cluster (id: {cluster_id}) and identify Digital Features.\n\n"
        f"Code nodes in this cluster:\n{node_lines}\n\n"
        "Identify only meaningful user-facing capabilities with real business value.\n"
        "Skip: loading screens, navigation bars, session handling, generic 'view X' screens, "
        "technical utilities, error states.\n"
        "If this cluster contains no real Digital Features, return [].\n\n"
        "Return ONLY a JSON array. Each element: name, description, user_value, "
        "confidence_score, business_capability_hint."
    )


def build_summary_prompt(features: list[dict]) -> str:
    """Build a user prompt to deduplicate and consolidate a list of extracted features.

    Args:
        features: List of already-extracted feature dicts.

    Returns:
        Prompt string requesting a consolidated JSON array.
    """
    features_json = json.dumps(features, indent=2)
    return (
        "You have been given a list of Digital Features extracted from multiple code clusters. "
        "Some may be duplicates, too vague, or not real Digital Features.\n\n"
        f"Features to review:\n{features_json}\n\n"
        "Tasks:\n"
        "1. REMOVE features that are not real Digital Features: loading screens, navigation chrome, "
        "session management, generic screens, technical plumbing.\n"
        "2. REMOVE features with confidence_score below 0.5.\n"
        "3. MERGE features that represent the same user-facing capability — keep the most descriptive version.\n"
        "4. Use the highest confidence_score among merged duplicates.\n"
        "5. Preserve business_capability_hint when available.\n\n"
        "Return ONLY a JSON array of high-quality Digital Feature objects, each with: "
        "name, description, user_value, confidence_score, business_capability_hint."
    )


def parse_llm_response(response: str) -> list[dict]:
    """Parse an LLM response string into a list of feature dicts.

    Strips markdown code fences if present before parsing JSON.

    Args:
        response: Raw string returned by the LLM.

    Returns:
        List of dicts representing Digital Features.

    Raises:
        ValueError: If the response cannot be parsed as a JSON array.
    """
    text = response.strip()

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        inner_lines = lines[1:]
        if inner_lines and inner_lines[-1].strip() == "```":
            inner_lines = inner_lines[:-1]
        text = "\n".join(inner_lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM response is not valid JSON. Parse error: {exc}\n"
            f"Response (first 500 chars): {response[:500]}"
        ) from exc

    if not isinstance(parsed, list):
        raise ValueError(
            f"Expected a JSON array from LLM, got {type(parsed).__name__}. "
            f"Response (first 500 chars): {response[:500]}"
        )

    return parsed
