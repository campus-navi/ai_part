from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.academic_plan import (
    count_korean_characters_impl,
    _load_academic_plan_skill_impl,
    _load_humanize_korean_skill_impl,
    _retrieve_reference_examples_impl,
)


mcp = FastMCP("academic-plan-rag")


@mcp.tool()
def mcp_load_academic_plan_skill(major_type: str, section_key: str) -> dict:
    """Return academic-plan revision skill guidance."""
    return _load_academic_plan_skill_impl(major_type, section_key)


@mcp.tool()
def mcp_load_humanize_korean_skill() -> dict:
    """Return Korean humanizing guidance for final style cleanup."""
    return _load_humanize_korean_skill_impl()


@mcp.tool()
def mcp_retrieve_reference_examples(
    target_name: str,
    user_department: str,
    section_key: str,
    limit: int = 2,
) -> list[dict]:
    """Retrieve academic-plan reference examples for RAG."""
    return _retrieve_reference_examples_impl(
        target_name=target_name,
        user_department=user_department,
        section_key=section_key,
        limit=limit,
    )


@mcp.tool()
def mcp_count_korean_characters(text: str) -> dict:
    """Count Korean application characters, excluding whitespace."""
    return count_korean_characters_impl(text)


if __name__ == "__main__":
    mcp.run()
