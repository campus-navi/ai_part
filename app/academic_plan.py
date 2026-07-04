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
- For every section, call the local skill and reference tools before writing the final answer.
- If mcp_* tools are available, use them as additional RAG evidence.
- Return Korean structured output only.
- Each section must include section_key, revised_content, revision reasons, and retrieved context.
- Do not generate original_content, diff, changes, or inline_diff. The server computes them.
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


class TextChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: TextChangeType
    original_text: str | None = None
    revised_text: str | None = None
    inline_diff: list[InlineDiffPart] = Field(default_factory=list)


class AcademicPlanAgentSectionRevision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_key: SectionKey
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
    changes: list[TextChange] = Field(default_factory=list)
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
                        temperature=0.2,
                        max_tokens=self.config.max_tokens,
                        parallel_tool_calls=True,
                    ),
                    tools=_build_tools(),
                    mcp_servers=mcp_servers,
                    output_type=AcademicPlanAgentOutput,
                )
                result = await Runner.run(agent, _format_agent_input(request))
        except AcademicPlanConfigurationError:
            raise
        except Exception as exc:
            raise AcademicPlanExecutionError(f"academic plan review failed: {exc}") from exc

        if result.final_output is None:
            raise AcademicPlanExecutionError("AI response did not include final output")
        try:
            output = AcademicPlanAgentOutput.model_validate(result.final_output)
            return _assemble_review_response(request, output)
        except AcademicPlanExecutionError:
            raise
        except Exception as exc:
            raise AcademicPlanExecutionError("AI response did not match expected schema") from exc


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

    return [load_academic_plan_skill, retrieve_reference_examples, build_section_diff]


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
                inline_diff=build_inline_diff_impl(original_text or "", revised_text or ""),
            )
        )

    return changes


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
                changes=build_text_changes_impl(original, section.revised_content),
                reasons=section.reasons,
                retrieved_context=section.retrieved_context,
            )
        )

    return AcademicPlanReviewResponse(
        metadata=request.metadata,
        sections=sections,
        overall_comment=agent_output.overall_comment,
    )


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
                    revised_content=section.revised_content,
                    reasons=section.reasons,
                    retrieved_context=section.retrieved_context,
                )
                for section in response.sections
            ],
            overall_comment=response.overall_comment,
        ),
    )


def _format_agent_input(request: AcademicPlanReviewRequest) -> str:
    return (
        "Revise this academic plan request. Use tools before final output.\n"
        f"{request.model_dump_json(indent=2)}"
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
