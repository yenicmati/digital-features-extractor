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
