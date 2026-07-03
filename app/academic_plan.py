from __future__ import annotations

import difflib
import os
import shlex
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
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


MAJOR_TYPE_LABELS: dict[str, str] = {
    "DOUBLE_MAJOR": "이중전공",
    "COMPLEX_MAJOR": "복합전공",
    "CONVERGENCE_MAJOR": "융합전공",
    "STUDENT_DESIGN": "학생설계전공",
}

SECTION_GUIDES: dict[str, dict[str, str]] = {
    "application_motive": {
        "title": "지원동기",
        "skill": "전공 선택의 계기, 기존 전공과의 연결, 목표 전공이 필요한 이유를 한 흐름으로 묶는다.",
        "checklist": "막연한 흥미보다 경험-문제의식-전공 필요성 순서로 구체화한다.",
    },
    "interest_field": {
        "title": "관심분야",
        "skill": "관심 주제를 전공 내 세부 분야와 연결하고, 탐구 질문이나 적용 장면을 제시한다.",
        "checklist": "키워드 나열을 피하고 왜 그 분야가 필요한지 설명한다.",
    },
    "study_plan": {
        "title": "학업계획",
        "skill": "수강, 프로젝트, 연구, 비교과 활동을 학기 또는 단계별 계획으로 정리한다.",
        "checklist": "과목명만 나열하지 말고 실행 방식과 산출물을 포함한다.",
    },
    "etc_info": {
        "title": "기타",
        "skill": "역량, 준비도, 향후 진로처럼 앞 섹션을 보완하는 정보를 중복 없이 배치한다.",
        "checklist": "자기소개 반복보다 선발자가 판단할 추가 근거를 담는다.",
    },
}

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
- Each section must include original_content, revised_content, a unified diff, and revision reasons.
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


class AcademicPlanSectionReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_key: SectionKey
    original_content: str
    revised_content: str
    diff: str = ""
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
                    output_type=AcademicPlanReviewResponse,
                )
                result = await Runner.run(agent, _format_agent_input(request))
        except AcademicPlanConfigurationError:
            raise
        except Exception as exc:
            raise AcademicPlanExecutionError(f"academic plan review failed: {exc}") from exc

        if result.final_output is None:
            raise AcademicPlanExecutionError("AI response did not include final output")
        try:
            output = AcademicPlanReviewResponse.model_validate(result.final_output)
            return _normalize_review_response(request, output)
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
    section = SECTION_GUIDES.get(section_key, SECTION_GUIDES["etc_info"])
    return {
        "source_type": "skill",
        "major_type": major_type,
        "major_type_label": MAJOR_TYPE_LABELS.get(major_type, major_type),
        "section_key": section_key,
        "section_title": section["title"],
        "skill": section["skill"],
        "checklist": section["checklist"],
        "common_rule": "전공 적합성, 구체성, 실행 가능성을 우선하고 사실을 새로 만들지 않는다.",
    }


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


def _normalize_review_response(
    request: AcademicPlanReviewRequest,
    response: AcademicPlanReviewResponse,
) -> AcademicPlanReviewResponse:
    originals = {section.section_key: section.content for section in request.sections}
    response_keys = {section.section_key for section in response.sections}
    request_keys = set(originals)
    if response_keys != request_keys:
        raise AcademicPlanExecutionError(
            f"AI response section mismatch: expected {sorted(request_keys)}, got {sorted(response_keys)}"
        )

    sections = []
    for section in response.sections:
        original = originals[section.section_key]
        sections.append(
            section.model_copy(
                update={
                    "original_content": original,
                    "diff": build_section_diff_impl(
                        section.section_key,
                        original,
                        section.revised_content,
                    ),
                }
            )
        )

    return response.model_copy(
        update={
            "document_type": "ACADEMIC_PLAN",
            "metadata": request.metadata,
            "sections": sections,
        }
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
