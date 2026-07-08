from __future__ import annotations

import difflib
import os
import re
import shlex
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


MajorType = Literal[
    "DOUBLE_MAJOR",
    "COMPLEX_MAJOR",
    "CONVERGENCE_MAJOR",
    "STUDENT_DESIGN",
]
SectionKey = Literal[
    "application_motive",
    "interest_field",
    "study_plan",
    "etc_info",
]
RevisionCategory = Literal[
    "clarity",
    "specificity",
    "fit",
    "structure",
    "evidence",
    "tone",
]
TextChangeType = Literal["replace", "insert", "delete"]
InlineDiffType = Literal["equal", "replace", "insert", "delete"]


MAJOR_TYPE_LABELS: dict[str, str] = {
    "DOUBLE_MAJOR": "이중전공",
    "COMPLEX_MAJOR": "복합전공",
    "CONVERGENCE_MAJOR": "융합전공",
    "STUDENT_DESIGN": "학생설계전공",
}

SECTION_TITLES: dict[str, str] = {
    "application_motive": "지원동기",
    "interest_field": "관심분야",
    "study_plan": "학업계획",
    "etc_info": "기타",
}
SKILL_PATH = Path(__file__).resolve().parent.parent / "skills" / "academic_plan_review" / "SKILL.md"
HUMANIZE_SKILL_PATH = Path(__file__).resolve().parent.parent / "skills" / "humanize_korean" / "SKILL.md"
HUMANIZE_QUICK_RULES_PATH = (
    Path(__file__).resolve().parent.parent / "skills" / "humanize_korean" / "references" / "quick-rules.md"
)

REFERENCE_EXAMPLES: tuple[dict[str, str], ...] = (
    {
        "section_key": "application_motive",
        "title": "컴퓨터공학-경제학 연결 지원동기",
        "content": "데이터 처리 경험에서 사회 현상을 정량적으로 이해할 필요를 느꼈고, 경제학의 이론과 실증 분석을 함께 배우려는 흐름이 설득력 있다.",
    },
    {
        "section_key": "interest_field",
        "title": "응용미시/데이터 분석 관심분야",
        "content": "플랫폼, 노동, 교육처럼 데이터가 축적되는 시장을 관심 분야로 잡고, 컴퓨터공학 역량을 계량 분석에 활용하겠다고 좁히면 좋다.",
    },
    {
        "section_key": "study_plan",
        "title": "단계형 학업계획",
        "content": "기초 미시/거시/통계 과목으로 언어를 익힌 뒤 계량경제, 산업조직, 프로젝트 과목으로 확장하는 순서가 읽기 쉽다.",
    },
    {
        "section_key": "etc_info",
        "title": "추가 역량 정리",
        "content": "프로그래밍, 통계, 협업 경험은 경제학 학습에서 어떻게 검증 가능한 산출물로 이어질지 짧게 제시한다.",
    },
)

ACADEMIC_PLAN_AGENT_INSTRUCTIONS = """
You are a Korean academic-plan revision agent.
Revise the user's academic plan by section, using retrieved revision skills,
reference examples, and MCP RAG tools when configured.

Rules:
- Preserve the user's facts. Do not invent awards, courses completed, internships, grades, or research.
- Improve specificity, fit to the target major, section structure, and admissions tone.
- For every section, call load_academic_plan_skill and retrieve_reference_examples before drafting.
- For every section, call load_humanize_korean_skill before finalizing.
- First create draft_content from the academic-plan skill/reference evidence.
- Then create revised_content by applying humanize-korean only to reduce AI-like Korean phrasing in draft_content.
- revised_content must preserve draft_content facts, argument order, register, and length policy.
- For every section, call count_korean_characters on the original and revised content before final output.
- If count_korean_characters shows revised_content is below the required minimum, expand the same section before final output.
- If mcp_* tools are available, use them as additional RAG evidence.
- Return Korean structured output only.
- Each section must include section_key, draft_content, revised_content, revision reasons, and retrieved context.
- Do not generate original_content, diff, suggestions, or preview_diff. The server computes them.
- Each revision reason must name the changed text, the reason, and the supporting skill/example/MCP evidence.
""".strip()


class AcademicPlanMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    major_type: MajorType
    target_name: str
    user_department: str


class AcademicPlanSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_key: SectionKey
    content: str


class AcademicPlanReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: Literal["ACADEMIC_PLAN"]
    metadata: AcademicPlanMetadata
    sections: list[AcademicPlanSection]


class RetrievedContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["skill", "example", "mcp"]
    title: str
    content: str


class RevisionReason(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_text: str | None = None
    revised_text: str | None = None
    category: RevisionCategory
    reason: str
    evidence: list[str] = Field(default_factory=list)


class InlineDiffPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: InlineDiffType
    text: str | None = None
    original_text: str | None = None
    revised_text: str | None = None


class TextRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: int
    end: int


class SuggestionReason(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: RevisionCategory
    summary: str


class TextSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suggestion_id: str
    type: TextChangeType
    original_range: TextRange
    original_text: str | None = None
    suggested_text: str | None = None
    reason: SuggestionReason
    preview_diff: list[InlineDiffPart] = Field(default_factory=list)


class TextChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: TextChangeType
    original_text: str | None = None
    revised_text: str | None = None
    inline_diff: list[InlineDiffPart] = Field(default_factory=list)


class AcademicPlanAgentSectionRevision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_key: SectionKey
    draft_content: str
    revised_content: str
    reasons: list[RevisionReason]
    retrieved_context: list[RetrievedContext] = Field(default_factory=list)


class AcademicPlanAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sections: list[AcademicPlanAgentSectionRevision]
    overall_comment: str


class AcademicPlanSectionReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_key: SectionKey
    original_content: str
    revised_content: str
    diff: str = ""
    suggestions: list[TextSuggestion] = Field(default_factory=list)
    reasons: list[RevisionReason]
    retrieved_context: list[RetrievedContext] = Field(default_factory=list)


class AcademicPlanReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: Literal["ACADEMIC_PLAN"] = "ACADEMIC_PLAN"
    metadata: AcademicPlanMetadata
    sections: list[AcademicPlanSectionReview]
    overall_comment: str


class AcademicPlanReviewer(Protocol):
    async def review(self, request: AcademicPlanReviewRequest) -> AcademicPlanReviewResponse:
        ...


class AcademicPlanConfigurationError(RuntimeError):
    pass


class AcademicPlanExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class AcademicPlanAgentConfig:
    model: str = "gpt-4.1-mini"
    max_tokens: int = 3000

    @classmethod
    def from_env(cls) -> AcademicPlanAgentConfig:
        return cls(
            model=os.getenv("ACADEMIC_PLAN_MODEL", cls.model),
            max_tokens=_env_int("ACADEMIC_PLAN_MAX_TOKENS", cls.max_tokens),
        )


class OpenAIAcademicPlanReviewer:
    def __init__(self, config: AcademicPlanAgentConfig | None = None) -> None:
        self.config = config or AcademicPlanAgentConfig.from_env()

    async def review(self, request: AcademicPlanReviewRequest) -> AcademicPlanReviewResponse:
        _validate_request(request)
        if not os.getenv("OPENAI_API_KEY"):
            raise AcademicPlanConfigurationError("OPENAI_API_KEY is required")

        try:
            from agents import Agent, ModelSettings, Runner
        except ImportError as exc:
            raise AcademicPlanConfigurationError("openai-agents package is not installed") from exc

        try:
            async with _mcp_servers_from_env() as mcp_servers:
                agent = Agent(
                    name="Academic Plan Revision Agent",
                    instructions=ACADEMIC_PLAN_AGENT_INSTRUCTIONS,
                    model=self.config.model,
                    model_settings=ModelSettings(
                        max_tokens=self.config.max_tokens,
                        parallel_tool_calls=True,
                    ),
                    tools=_build_tools(),
                    mcp_servers=mcp_servers,
                    output_type=AcademicPlanAgentOutput,
                )
                section_outputs = []
                for section in request.sections:
                    section_outputs.append(
                        await _run_section_agent(
                            Runner,
                            agent,
                            AcademicPlanReviewRequest(
                                document_type=request.document_type,
                                metadata=request.metadata,
                                sections=[section],
                            ),
                        )
                    )
        except AcademicPlanConfigurationError:
            raise
        except Exception as exc:
            raise AcademicPlanExecutionError(f"academic plan review failed: {exc}") from exc

        return _assemble_review_response(
            request,
            AcademicPlanAgentOutput(
                sections=section_outputs,
                overall_comment=_build_overall_comment(section_outputs),
            ),
        )


async def _run_section_agent(
    runner: Any,
    agent: Any,
    request: AcademicPlanReviewRequest,
) -> AcademicPlanAgentSectionRevision:
    expected_key = request.sections[0].section_key
    original = request.sections[0].content
    retry_reason: str | None = None

    for _ in range(2):
        result = await runner.run(agent, _format_agent_input(request, retry_reason))
        if result.final_output is None:
            retry_reason = "AI response did not include final output"
            continue
        try:
            output = AcademicPlanAgentOutput.model_validate(result.final_output)
        except Exception as exc:
            retry_reason = f"AI response did not match expected schema: {exc}"
            continue

        matching = [section for section in output.sections if section.section_key == expected_key]
        if len(matching) != 1:
            retry_reason = (
                f"AI response section mismatch: expected [{expected_key}], "
                f"got {[section.section_key for section in output.sections]}"
            )
            continue

        try:
            _validate_revision_length(expected_key, original, matching[0].revised_content)
        except AcademicPlanExecutionError as exc:
            retry_reason = str(exc)
            continue
        return matching[0]

    raise AcademicPlanExecutionError(retry_reason or "AI response did not include a valid section")


def _build_overall_comment(sections: list[AcademicPlanAgentSectionRevision]) -> str:
    titles = ", ".join(SECTION_TITLES.get(section.section_key, section.section_key) for section in sections)
    return f"{titles} 섹션을 첨삭 기준에 따라 보완했습니다."


def _validate_request(request: AcademicPlanReviewRequest) -> None:
    if not request.sections:
        raise ValueError("sections must include at least one item")
    seen: set[str] = set()
    for section in request.sections:
        if section.section_key in seen:
            raise ValueError(f"duplicate section_key: {section.section_key}")
        seen.add(section.section_key)
        if not section.content.strip():
            raise ValueError(f"{section.section_key} content is required")


def _build_tools() -> list[Any]:
    from agents import function_tool

    @function_tool
    def load_academic_plan_skill(major_type: str, section_key: str) -> dict[str, Any]:
        """Return the revision skill for a major type and academic-plan section."""
        return _load_academic_plan_skill_impl(major_type, section_key)

    @function_tool
    def load_humanize_korean_skill() -> dict[str, Any]:
        """Return Korean humanizing guidance for final style cleanup."""
        return _load_humanize_korean_skill_impl()

    @function_tool
    def retrieve_reference_examples(
        target_name: str,
        user_department: str,
        section_key: str,
        limit: int = 2,
    ) -> list[dict[str, str]]:
        """Retrieve concise reference examples for section-level RAG."""
        return _retrieve_reference_examples_impl(
            target_name=target_name,
            user_department=user_department,
            section_key=section_key,
            limit=limit,
        )

    @function_tool
    def build_section_diff(section_key: str, original_content: str, revised_content: str) -> str:
        """Build a unified diff between the original and revised section."""
        return build_section_diff_impl(section_key, original_content, revised_content)

    @function_tool
    def count_korean_characters(text: str) -> dict[str, int]:
        """Count Korean application characters, excluding whitespace."""
        return count_korean_characters_impl(text)

    return [
        load_academic_plan_skill,
        load_humanize_korean_skill,
        retrieve_reference_examples,
        build_section_diff,
        count_korean_characters,
    ]


def _load_academic_plan_skill_impl(major_type: str, section_key: str) -> dict[str, Any]:
    sections = _load_academic_plan_skill_sections()
    return {
        "source_type": "skill",
        "source_path": str(SKILL_PATH),
        "major_type": major_type,
        "major_type_label": MAJOR_TYPE_LABELS.get(major_type, major_type),
        "section_key": section_key,
        "section_title": SECTION_TITLES.get(section_key, section_key),
        "skill": sections.get(section_key, ""),
        "checklist": sections.get("common", ""),
        "common_rule": "전공 적합성, 구체성, 실행 가능성을 우선하고 사실을 새로 만들지 않는다.",
    }


def _load_humanize_korean_skill_impl() -> dict[str, Any]:
    return {
        "source_type": "skill",
        "source_path": str(HUMANIZE_SKILL_PATH),
        "quick_rules_path": str(HUMANIZE_QUICK_RULES_PATH),
        "skill": HUMANIZE_SKILL_PATH.read_text(encoding="utf-8"),
        "quick_rules": HUMANIZE_QUICK_RULES_PATH.read_text(encoding="utf-8"),
    }


@lru_cache(maxsize=1)
def _load_academic_plan_skill_sections() -> dict[str, str]:
    text = SKILL_PATH.read_text(encoding="utf-8")
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for line in text.splitlines():
        match = re.match(r"^##\s+([a-z_]+)\s*$", line)
        if match:
            current = match.group(1)
            sections[current] = []
            continue
        if current:
            sections[current].append(line)

    return {key: "\n".join(lines).strip() for key, lines in sections.items()}


def _retrieve_reference_examples_impl(
    target_name: str,
    user_department: str,
    section_key: str,
    limit: int = 2,
) -> list[dict[str, str]]:
    query = f"{target_name} {user_department} {section_key}".lower()
    scored: list[tuple[int, dict[str, str]]] = []
    for example in REFERENCE_EXAMPLES:
        score = 2 if example["section_key"] == section_key else 0
        score += sum(1 for token in query.split() if token and token in example["content"].lower())
        scored.append((score, example))

    return [
        {
            "source_type": "example",
            "title": example["title"],
            "content": example["content"],
        }
        for _, example in sorted(scored, key=lambda item: item[0], reverse=True)[: max(limit, 1)]
    ]


def build_section_diff_impl(section_key: str, original_content: str, revised_content: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            original_content.splitlines(),
            revised_content.splitlines(),
            fromfile=f"{section_key}.original",
            tofile=f"{section_key}.revised",
            lineterm="",
        )
    )


def count_korean_characters_impl(text: str) -> dict[str, int]:
    return {
        "characters": len(text),
        "characters_without_whitespace": len(re.sub(r"\s+", "", text)),
    }


def build_text_changes_impl(original_content: str, revised_content: str) -> list[TextChange]:
    original_units = _split_revision_units(original_content)
    revised_units = _split_revision_units(revised_content)
    matcher = difflib.SequenceMatcher(None, original_units, revised_units)
    changes: list[TextChange] = []

    for tag, original_start, original_end, revised_start, revised_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        original_text = "\n\n".join(original_units[original_start:original_end]) or None
        revised_text = "\n\n".join(revised_units[revised_start:revised_end]) or None
        changes.append(
            TextChange(
                type=tag,
                original_text=original_text,
                revised_text=revised_text,
                inline_diff=(
                    build_inline_diff_impl(original_text or "", revised_text or "")
                    if _should_show_inline_diff(original_text or "", revised_text or "")
                    else []
                ),
            )
        )

    return changes


def build_text_suggestions_impl(
    section_key: str,
    original_content: str,
    revised_content: str,
    reasons: list[RevisionReason],
) -> list[TextSuggestion]:
    if original_content == revised_content:
        return []

    original_units = _split_revision_units_with_ranges(original_content)
    revised_units = _split_revision_units_with_ranges(revised_content)
    if len(original_units) == len(revised_units):
        suggestions = []
        for original_unit, revised_unit in zip(original_units, revised_units, strict=True):
            if original_unit[0] == revised_unit[0]:
                continue
            _append_text_suggestion(
                suggestions=suggestions,
                section_key=section_key,
                change_type="replace",
                original_range=TextRange(start=original_unit[1], end=original_unit[2]),
                original_text=original_unit[0],
                suggested_text=revised_unit[0],
                reasons=reasons,
            )
        if _apply_text_suggestions(original_content, suggestions) == revised_content:
            return suggestions

    matcher = difflib.SequenceMatcher(
        None,
        [unit[0] for unit in original_units],
        [unit[0] for unit in revised_units],
    )
    suggestions: list[TextSuggestion] = []

    for tag, original_start, original_end, revised_start, revised_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            _append_replacement_chunk_suggestions(
                suggestions=suggestions,
                section_key=section_key,
                original_content=original_content,
                revised_content=revised_content,
                original_units=original_units,
                revised_units=revised_units,
                original_start=original_start,
                original_end=original_end,
                revised_start=revised_start,
                revised_end=revised_end,
                reasons=reasons,
            )
            continue

        original_range, revised_range = _change_ranges(
            tag,
            original_units,
            revised_units,
            original_start,
            original_end,
            revised_start,
            revised_end,
            len(original_content),
            len(revised_content),
        )
        _append_text_suggestion(
            suggestions=suggestions,
            section_key=section_key,
            change_type=tag,
            original_range=TextRange(start=original_range[0], end=original_range[1]),
            original_text=original_content[original_range[0] : original_range[1]] or None,
            suggested_text=revised_content[revised_range[0] : revised_range[1]] or None,
            reasons=reasons,
        )

    if _apply_text_suggestions(original_content, suggestions) != revised_content:
        return [
            _make_text_suggestion(
                section_key=section_key,
                index=1,
                change_type="replace",
                original_range=TextRange(start=0, end=len(original_content)),
                original_text=original_content,
                suggested_text=revised_content,
                reasons=reasons,
            )
        ]
    return suggestions


def _append_replacement_chunk_suggestions(
    *,
    suggestions: list[TextSuggestion],
    section_key: str,
    original_content: str,
    revised_content: str,
    original_units: list[tuple[str, int, int]],
    revised_units: list[tuple[str, int, int]],
    original_start: int,
    original_end: int,
    revised_start: int,
    revised_end: int,
    reasons: list[RevisionReason],
) -> None:
    original_count = original_end - original_start
    revised_count = revised_end - revised_start
    pair_count = min(original_count, revised_count)

    for offset in range(pair_count):
        original_range = _unit_range_with_separator(
            original_units,
            original_start + offset,
            len(original_content),
        )
        revised_range = _unit_range_with_separator(
            revised_units,
            revised_start + offset,
            len(revised_content),
        )
        _append_text_suggestion(
            suggestions=suggestions,
            section_key=section_key,
            change_type="replace",
            original_range=TextRange(start=original_range[0], end=original_range[1]),
            original_text=original_content[original_range[0] : original_range[1]],
            suggested_text=revised_content[revised_range[0] : revised_range[1]],
            reasons=reasons,
        )

    if original_count > pair_count:
        original_range = (
            original_units[original_start + pair_count][1],
            _unit_range_with_separator(original_units, original_end - 1, len(original_content))[1],
        )
        _append_text_suggestion(
            suggestions=suggestions,
            section_key=section_key,
            change_type="delete",
            original_range=TextRange(start=original_range[0], end=original_range[1]),
            original_text=original_content[original_range[0] : original_range[1]],
            suggested_text=None,
            reasons=reasons,
        )

    if revised_count > pair_count:
        insertion_point = (
            _unit_range_with_separator(original_units, original_start + pair_count - 1, len(original_content))[1]
            if pair_count
            else original_units[original_start][1]
        )
        revised_range = (
            revised_units[revised_start + pair_count][1],
            _unit_range_with_separator(revised_units, revised_end - 1, len(revised_content))[1],
        )
        _append_text_suggestion(
            suggestions=suggestions,
            section_key=section_key,
            change_type="insert",
            original_range=TextRange(start=insertion_point, end=insertion_point),
            original_text=None,
            suggested_text=revised_content[revised_range[0] : revised_range[1]],
            reasons=reasons,
        )


def _unit_range_with_separator(
    units: list[tuple[str, int, int]],
    index: int,
    text_len: int,
) -> tuple[int, int]:
    start = units[index][1]
    if index + 1 < len(units):
        return start, units[index + 1][1]
    return start, text_len


def _append_text_suggestion(
    *,
    suggestions: list[TextSuggestion],
    section_key: str,
    change_type: TextChangeType,
    original_range: TextRange,
    original_text: str | None,
    suggested_text: str | None,
    reasons: list[RevisionReason],
) -> None:
    suggestions.append(
        _make_text_suggestion(
            section_key=section_key,
            index=len(suggestions) + 1,
            change_type=change_type,
            original_range=original_range,
            original_text=original_text,
            suggested_text=suggested_text,
            reasons=reasons,
        )
    )


def _make_text_suggestion(
    *,
    section_key: str,
    index: int,
    change_type: TextChangeType,
    original_range: TextRange,
    original_text: str | None,
    suggested_text: str | None,
    reasons: list[RevisionReason],
) -> TextSuggestion:
    return TextSuggestion(
        suggestion_id=f"{section_key}-{index:03d}",
        type=change_type,
        original_range=original_range,
        original_text=original_text,
        suggested_text=suggested_text,
        reason=_suggestion_reason(original_text, suggested_text, reasons),
        preview_diff=(
            build_inline_diff_impl(original_text or "", suggested_text or "")
            if _should_show_inline_diff(original_text or "", suggested_text or "")
            else []
        ),
    )


def _apply_text_suggestions(text: str, suggestions: list[TextSuggestion]) -> str:
    result = text
    for suggestion in sorted(suggestions, key=lambda item: item.original_range.start, reverse=True):
        replacement = suggestion.suggested_text or ""
        result = result[: suggestion.original_range.start] + replacement + result[suggestion.original_range.end :]
    return result


def _suggestion_reason(
    original_text: str | None,
    suggested_text: str | None,
    reasons: list[RevisionReason],
) -> SuggestionReason:
    for reason in reasons:
        if _reason_matches_text(reason, original_text, suggested_text):
            return SuggestionReason(category=reason.category, summary=reason.reason)
    if reasons:
        return SuggestionReason(category=reasons[0].category, summary=reasons[0].reason)
    return SuggestionReason(category="structure", summary="문장 구조와 표현을 개선했습니다.")


def _reason_matches_text(
    reason: RevisionReason,
    original_text: str | None,
    suggested_text: str | None,
) -> bool:
    return _text_overlaps(original_text, reason.original_text) or _text_overlaps(
        suggested_text,
        reason.revised_text,
    )


def _text_overlaps(text: str | None, snippet: str | None) -> bool:
    if not text or not snippet:
        return False
    if snippet in text or text in snippet:
        return True
    compact_text = re.sub(r"\s+", "", text)
    compact_snippet = re.sub(r"\s+", "", snippet)
    if len(compact_snippet) < 8 or not compact_text:
        return False
    return compact_snippet[:24] in compact_text or compact_text[:24] in compact_snippet


def _change_ranges(
    tag: str,
    original_units: list[tuple[str, int, int]],
    revised_units: list[tuple[str, int, int]],
    original_start: int,
    original_end: int,
    revised_start: int,
    revised_end: int,
    original_len: int,
    revised_len: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if tag == "insert":
        original_point = _insertion_point(original_units, original_start, original_len)
        revised_range = _inserted_range(revised_units, revised_start, revised_end, revised_len)
        return (original_point, original_point), revised_range
    if tag == "delete":
        original_range = _deleted_range(original_units, original_start, original_end, original_len)
        return original_range, (0, 0)
    return (
        (original_units[original_start][1], original_units[original_end - 1][2]),
        (revised_units[revised_start][1], revised_units[revised_end - 1][2]),
    )


def _insertion_point(units: list[tuple[str, int, int]], index: int, text_len: int) -> int:
    if index > 0:
        return units[index - 1][2]
    if index < len(units):
        return units[index][1]
    return text_len


def _inserted_range(
    revised_units: list[tuple[str, int, int]],
    revised_start: int,
    revised_end: int,
    revised_len: int,
) -> tuple[int, int]:
    if revised_start > 0:
        start = revised_units[revised_start - 1][2]
    else:
        start = revised_units[revised_start][1]
    if revised_start == 0 and revised_end < len(revised_units):
        end = revised_units[revised_end][1]
    else:
        end = revised_units[revised_end - 1][2] if revised_end > revised_start else revised_len
    return start, end


def _deleted_range(
    original_units: list[tuple[str, int, int]],
    original_start: int,
    original_end: int,
    original_len: int,
) -> tuple[int, int]:
    if original_start > 0:
        start = original_units[original_start - 1][2]
    else:
        start = original_units[original_start][1]
    if original_start == 0 and original_end < len(original_units):
        end = original_units[original_end][1]
    else:
        end = original_units[original_end - 1][2] if original_end > original_start else original_len
    return start, end


def _should_show_inline_diff(original_text: str, revised_text: str) -> bool:
    max_len = max(len(original_text), len(revised_text))
    if max_len <= 240:
        return True
    ratio = difflib.SequenceMatcher(None, original_text, revised_text).ratio()
    length_gap = abs(len(original_text) - len(revised_text)) / max_len
    return ratio >= 0.82 and length_gap <= 0.25


def build_inline_diff_impl(original_text: str, revised_text: str) -> list[InlineDiffPart]:
    matcher = difflib.SequenceMatcher(None, original_text, revised_text)
    parts: list[InlineDiffPart] = []

    for tag, original_start, original_end, revised_start, revised_end in matcher.get_opcodes():
        original_part = original_text[original_start:original_end]
        revised_part = revised_text[revised_start:revised_end]
        if tag == "equal":
            parts.append(InlineDiffPart(type="equal", text=original_part))
        elif tag == "replace":
            parts.append(
                InlineDiffPart(
                    type="replace",
                    original_text=original_part,
                    revised_text=revised_part,
                )
            )
        elif tag == "delete":
            parts.append(InlineDiffPart(type="delete", original_text=original_part))
        elif tag == "insert":
            parts.append(InlineDiffPart(type="insert", revised_text=revised_part))

    return parts


def _split_revision_units(text: str) -> list[str]:
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = [
            item.strip()
            for item in re.findall(r".+?(?:[.!?。！？](?:\s+|$)|$)", paragraph, flags=re.S)
            if item.strip()
        ]
        units.extend(sentences or [paragraph])
    return units or [text]


def _split_revision_units_with_ranges(text: str) -> list[tuple[str, int, int]]:
    units_with_ranges: list[tuple[str, int, int]] = []
    cursor = 0
    for unit in _split_revision_units(text):
        start = text.find(unit, cursor)
        if start == -1:
            start = text.find(unit)
        if start == -1:
            continue
        end = start + len(unit)
        units_with_ranges.append((unit, start, end))
        cursor = end
    return units_with_ranges or [(text, 0, len(text))]


def _assemble_review_response(
    request: AcademicPlanReviewRequest,
    agent_output: AcademicPlanAgentOutput,
) -> AcademicPlanReviewResponse:
    originals = {section.section_key: section.content for section in request.sections}
    response_keys = {section.section_key for section in agent_output.sections}
    request_keys = set(originals)
    if response_keys != request_keys:
        raise AcademicPlanExecutionError(
            f"AI response section mismatch: expected {sorted(request_keys)}, got {sorted(response_keys)}"
        )

    sections = []
    for section in agent_output.sections:
        original = originals[section.section_key]
        _validate_revision_length(section.section_key, original, section.revised_content)
        sections.append(
            AcademicPlanSectionReview(
                section_key=section.section_key,
                original_content=original,
                revised_content=section.revised_content,
                diff=build_section_diff_impl(
                    section.section_key,
                    original,
                    section.revised_content,
                ),
                suggestions=build_text_suggestions_impl(
                    section.section_key,
                    original,
                    section.revised_content,
                    section.reasons,
                ),
                reasons=section.reasons,
                retrieved_context=section.retrieved_context,
            )
        )

    return AcademicPlanReviewResponse(
        metadata=request.metadata,
        sections=sections,
        overall_comment=agent_output.overall_comment,
    )


def _validate_revision_length(section_key: str, original_content: str, revised_content: str) -> None:
    original_len = len(original_content)
    revised_len = len(revised_content)
    minimum = _minimum_revised_length(original_len)
    if minimum and revised_len < minimum:
        raise AcademicPlanExecutionError(
            f"{section_key} revised_content is too short: original={original_len}, revised={revised_len}, minimum={minimum}"
        )
    if revised_len > 1000:
        raise AcademicPlanExecutionError(
            f"{section_key} revised_content exceeds 1000 characters: revised={revised_len}"
        )


def _minimum_revised_length(original_len: int) -> int | None:
    if original_len < 700:
        return None
    return max(650, int(original_len * 0.8))


def _normalize_review_response(
    request: AcademicPlanReviewRequest,
    response: AcademicPlanReviewResponse,
) -> AcademicPlanReviewResponse:
    return _assemble_review_response(
        request,
        AcademicPlanAgentOutput(
            sections=[
                AcademicPlanAgentSectionRevision(
                    section_key=section.section_key,
                    draft_content=section.revised_content,
                    revised_content=section.revised_content,
                    reasons=section.reasons,
                    retrieved_context=section.retrieved_context,
                )
                for section in response.sections
            ],
            overall_comment=response.overall_comment,
        ),
    )


def _format_agent_input(
    request: AcademicPlanReviewRequest,
    retry_reason: str | None = None,
) -> str:
    section = request.sections[0] if len(request.sections) == 1 else None
    length_policy = ""
    if section:
        original_len = len(section.content)
        minimum = _minimum_revised_length(original_len)
        if minimum:
            length_policy = (
                "\nLength policy for this section:\n"
                f"- original characters: {original_len}\n"
                f"- revised_content must be at least {minimum} characters and at most 1000 characters.\n"
                "- Use count_korean_characters on original content and revised_content before final output.\n"
            )
    retry_policy = f"\nPrevious attempt failed: {retry_reason}\nFix only that issue.\n" if retry_reason else ""
    return (
        "Revise this academic plan request. Use tools before final output.\n"
        "Output draft_content before load_humanize_korean_skill is applied, "
        "then revised_content after it is applied.\n"
        f"{request.model_dump_json(indent=2)}"
        f"{length_policy}"
        f"{retry_policy}"
    )


@asynccontextmanager
async def _mcp_servers_from_env():
    try:
        from agents.mcp import MCPServerStdio
    except ImportError as exc:
        raise AcademicPlanConfigurationError("Agents SDK MCP support is not installed") from exc

    command = os.getenv("ACADEMIC_PLAN_MCP_COMMAND", sys.executable)
    args = shlex.split(os.getenv("ACADEMIC_PLAN_MCP_ARGS", "-m app.academic_plan_mcp"))
    async with MCPServerStdio(
        name=os.getenv("ACADEMIC_PLAN_MCP_NAME", "academic-plan-rag"),
        params={"command": command, "args": args},
        cache_tools_list=True,
    ) as server:
        yield [server]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default
