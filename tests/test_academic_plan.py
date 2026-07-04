import pytest
from fastapi.testclient import TestClient

from app.academic_plan import (
    AcademicPlanAgentOutput,
    AcademicPlanAgentSectionRevision,
    AcademicPlanMetadata,
    AcademicPlanReviewRequest,
    AcademicPlanReviewResponse,
    AcademicPlanSectionReview,
    RevisionReason,
    _assemble_review_response,
    _format_agent_input,
    _load_academic_plan_skill_impl,
    _load_humanize_korean_skill_impl,
    _normalize_review_response,
    _retrieve_reference_examples_impl,
    _run_section_agent,
    build_inline_diff_impl,
    build_section_diff_impl,
    build_text_changes_impl,
    count_korean_characters_impl,
)
from app.main import app, get_academic_plan_reviewer


class FakeAcademicPlanReviewer:
    def __init__(self) -> None:
        self.requests: list[AcademicPlanReviewRequest] = []

    async def review(self, request: AcademicPlanReviewRequest) -> AcademicPlanReviewResponse:
        self.requests.append(request)
        return AcademicPlanReviewResponse(
            metadata=request.metadata,
            sections=[
                AcademicPlanSectionReview(
                    section_key="application_motive",
                    original_content=request.sections[0].content,
                    revised_content="컴퓨터공학에서 익힌 데이터 분석 경험을 경제학의 의사결정 문제와 연결하고 싶습니다.",
                    diff="",
                    reasons=[
                        RevisionReason(
                            original_text=request.sections[0].content,
                            revised_text="데이터 분석 경험을 경제학의 의사결정 문제와 연결",
                            category="fit",
                            reason="지원 전공과 기존 전공의 연결을 분명히 했습니다.",
                            evidence=["지원동기 첨삭 스킬"],
                        )
                    ],
                )
            ],
            overall_comment="전공 간 연결은 분명하며, 학업계획에서 실행 단계를 더 보강하면 좋습니다.",
        )


def test_academic_plan_endpoint_returns_structured_review():
    reviewer = FakeAcademicPlanReviewer()
    app.dependency_overrides[get_academic_plan_reviewer] = lambda: reviewer
    try:
        response = TestClient(app).post(
            "/ai/academic-plan/review",
            json={
                "document_type": "ACADEMIC_PLAN",
                "metadata": {
                    "major_type": "DOUBLE_MAJOR",
                    "target_name": "경제학과",
                    "user_department": "컴퓨터공학과",
                },
                "sections": [
                    {
                        "section_key": "application_motive",
                        "content": "경제학을 배우고 싶습니다.",
                    }
                ],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["document_type"] == "ACADEMIC_PLAN"
    assert data["metadata"]["major_type"] == "DOUBLE_MAJOR"
    assert data["sections"][0]["section_key"] == "application_motive"
    assert data["sections"][0]["reasons"][0]["category"] == "fit"
    assert reviewer.requests[0].metadata.target_name == "경제학과"


def test_academic_plan_tools_return_skill_and_examples():
    skill = _load_academic_plan_skill_impl("DOUBLE_MAJOR", "study_plan")
    humanize_skill = _load_humanize_korean_skill_impl()
    examples = _retrieve_reference_examples_impl("경제학과", "컴퓨터공학과", "study_plan")

    assert skill["major_type_label"] == "이중전공"
    assert "학업계획" in skill["section_title"]
    assert "졸업 사정표" in skill["skill"]
    assert "가독성" in skill["checklist"]
    assert humanize_skill["source_type"] == "skill"
    assert "Humanize Korean" in humanize_skill["skill"]
    assert "Quick Rules" in humanize_skill["quick_rules"]
    assert examples[0]["source_type"] == "example"
    assert "과목" in examples[0]["content"]


def test_count_korean_characters_counts_total_and_non_whitespace():
    counts = count_korean_characters_impl("경제학을 체계적으로\n공부하겠습니다.")

    assert counts == {
        "characters": 19,
        "characters_without_whitespace": 17,
    }


def test_normalize_review_response_fills_original_content_and_diff():
    request = AcademicPlanReviewRequest(
        document_type="ACADEMIC_PLAN",
        metadata=AcademicPlanMetadata(
            major_type="DOUBLE_MAJOR",
            target_name="경제학과",
            user_department="컴퓨터공학과",
        ),
        sections=[
            {
                "section_key": "study_plan",
                "content": "경제학 과목을 듣겠습니다.",
            }
        ],
    )
    response = AcademicPlanReviewResponse(
        metadata=request.metadata,
        sections=[
            AcademicPlanSectionReview(
                section_key="study_plan",
                original_content="agent supplied text is ignored",
                revised_content="미시경제학과 통계학을 먼저 수강한 뒤 계량경제 프로젝트로 확장하겠습니다.",
                reasons=[
                    RevisionReason(
                        category="specificity",
                        reason="학습 순서와 산출물을 구체화했습니다.",
                    )
                ],
            )
        ],
        overall_comment="구체성이 개선되었습니다.",
    )

    normalized = _normalize_review_response(request, response)

    section = normalized.sections[0]
    assert section.original_content == "경제학 과목을 듣겠습니다."
    assert "--- study_plan.original" in section.diff
    assert "-경제학 과목을 듣겠습니다." in section.diff
    assert "+미시경제학과 통계학을 먼저 수강한 뒤 계량경제 프로젝트로 확장하겠습니다." in section.diff
    assert section.changes[0].type == "replace"
    assert section.changes[0].original_text == "경제학 과목을 듣겠습니다."


def test_assemble_review_response_uses_smaller_agent_output():
    request = AcademicPlanReviewRequest(
        document_type="ACADEMIC_PLAN",
        metadata=AcademicPlanMetadata(
            major_type="DOUBLE_MAJOR",
            target_name="경제학과",
            user_department="컴퓨터공학과",
        ),
        sections=[
            {
                "section_key": "application_motive",
                "content": "경제학을 매우 열심히 공부하겠습니다.",
            }
        ],
    )
    output = AcademicPlanAgentOutput(
        sections=[
            AcademicPlanAgentSectionRevision(
                section_key="application_motive",
                draft_content="경제학을 체계적으로 공부하겠습니다.",
                revised_content="경제학을 체계적으로 공부하겠습니다.",
                reasons=[
                    RevisionReason(
                        category="specificity",
                        reason="추상적 태도 표현을 학업 방식으로 구체화했습니다.",
                    )
                ],
            )
        ],
        overall_comment="표현이 구체화되었습니다.",
    )

    response = _assemble_review_response(request, output)
    section = response.sections[0]

    assert section.original_content == "경제학을 매우 열심히 공부하겠습니다."
    assert section.revised_content == "경제학을 체계적으로 공부하겠습니다."
    assert section.diff
    assert section.changes[0].inline_diff
    assert any(part.type == "replace" for part in section.changes[0].inline_diff)


def test_assemble_review_response_records_skill_stage_snapshots():
    request = AcademicPlanReviewRequest(
        document_type="ACADEMIC_PLAN",
        metadata=AcademicPlanMetadata(
            major_type="DOUBLE_MAJOR",
            target_name="경제학과",
            user_department="컴퓨터공학과",
        ),
        sections=[
            {
                "section_key": "application_motive",
                "content": "경제학을 매우 열심히 공부하겠습니다.",
            }
        ],
    )
    output = AcademicPlanAgentOutput(
        sections=[
            AcademicPlanAgentSectionRevision(
                section_key="application_motive",
                draft_content="경제학을 체계적으로 학습하겠습니다.",
                revised_content="경제학을 체계적으로 공부하겠습니다.",
                reasons=[
                    RevisionReason(
                        category="specificity",
                        reason="학습 방향을 구체화했습니다.",
                    )
                ],
            )
        ],
        overall_comment="표현이 구체화되었습니다.",
    )

    section = _assemble_review_response(request, output).sections[0]

    assert [stage.skill_name for stage in section.revision_stages] == [
        "academic_plan_review",
        "humanize_korean",
    ]
    assert section.revision_stages[0].before_content == request.sections[0].content
    assert section.revision_stages[0].after_content == "경제학을 체계적으로 학습하겠습니다."
    assert section.revision_stages[1].before_content == "경제학을 체계적으로 학습하겠습니다."
    assert section.revision_stages[1].after_content == section.revised_content


def test_assemble_review_response_rejects_overly_short_long_section_revision():
    original = "경제학과 컴퓨터공학을 연결해 학습하겠습니다. " * 30
    request = AcademicPlanReviewRequest(
        document_type="ACADEMIC_PLAN",
        metadata=AcademicPlanMetadata(
            major_type="DOUBLE_MAJOR",
            target_name="경제학과",
            user_department="컴퓨터공학과",
        ),
        sections=[
            {
                "section_key": "application_motive",
                "content": original,
            }
        ],
    )
    output = AcademicPlanAgentOutput(
        sections=[
            AcademicPlanAgentSectionRevision(
                section_key="application_motive",
                draft_content="경제학을 공부하겠습니다.",
                revised_content="경제학을 공부하겠습니다.",
                reasons=[RevisionReason(category="clarity", reason="짧게 정리했습니다.")],
            )
        ],
        overall_comment="짧습니다.",
    )

    with pytest.raises(Exception, match="too short|below 80%"):
        _assemble_review_response(request, output)


@pytest.mark.anyio
async def test_run_section_agent_retries_when_revision_is_too_short():
    original = "경제학과 컴퓨터공학을 연결해 학습하겠습니다. " * 30
    request = AcademicPlanReviewRequest(
        document_type="ACADEMIC_PLAN",
        metadata=AcademicPlanMetadata(
            major_type="DOUBLE_MAJOR",
            target_name="경제학과",
            user_department="컴퓨터공학과",
        ),
        sections=[
            {
                "section_key": "application_motive",
                "content": original,
            }
        ],
    )

    class FakeRunner:
        calls = 0

        @classmethod
        async def run(cls, agent, prompt):
            cls.calls += 1
            revised = (
                "경제학을 공부하겠습니다."
                if cls.calls == 1
                else "경제학과 컴퓨터공학을 연결해 학습하겠습니다. " * 26
            )
            return type(
                "Result",
                (),
                {
                    "final_output": AcademicPlanAgentOutput(
                        sections=[
                            AcademicPlanAgentSectionRevision(
                                section_key="application_motive",
                                draft_content=revised,
                                revised_content=revised,
                                reasons=[
                                    RevisionReason(
                                        category="clarity",
                                        reason="분량 기준에 맞춰 보완했습니다.",
                                    )
                                ],
                            )
                        ],
                        overall_comment="보완했습니다.",
                    )
                },
            )()

    section = await _run_section_agent(FakeRunner, object(), request)

    assert FakeRunner.calls == 2
    assert len(section.revised_content) >= int(len(original) * 0.8)


def test_format_agent_input_includes_length_policy_and_retry_reason():
    request = AcademicPlanReviewRequest(
        document_type="ACADEMIC_PLAN",
        metadata=AcademicPlanMetadata(
            major_type="DOUBLE_MAJOR",
            target_name="경제학과",
            user_department="컴퓨터공학과",
        ),
        sections=[
            {
                "section_key": "application_motive",
                "content": "경제학과 컴퓨터공학을 연결해 학습하겠습니다. " * 30,
            }
        ],
    )

    prompt = _format_agent_input(request, "too short")

    assert "revised_content must be at least" in prompt
    assert "Previous attempt failed: too short" in prompt
    assert "draft_content" in prompt
    assert "load_humanize_korean_skill" in prompt


def test_inline_diff_marks_only_changed_phrase():
    diff = build_inline_diff_impl(
        "경제학을 매우 열심히 공부하겠습니다.",
        "경제학을 체계적으로 공부하겠습니다.",
    )

    changed = [part for part in diff if part.type == "replace"]
    assert changed[0].original_text == "매우 열심히"
    assert changed[0].revised_text == "체계적으로"


def test_text_changes_skip_inline_diff_for_large_rewrites():
    original = (
        "컴퓨터공학 전공에서 자료구조와 데이터 분석을 배우며 기술이 사회 현상을 설명하는 데 유용하다는 점을 체감했습니다. "
        "하지만 서비스 이용자의 선택과 플랫폼 정책의 효과를 해석하려면 경제학 지식이 필요했습니다. "
    ) * 4
    revised = (
        "컴퓨터공학에서 익힌 데이터 처리 역량을 시장 참여자의 선택을 해석하는 분석 역량으로 확장하고자 합니다. "
        "경제학과 이중전공에서는 미시경제학과 계량경제학을 바탕으로 플랫폼 시장의 경쟁 구조를 공부하겠습니다. "
    ) * 3

    changes = build_text_changes_impl(original, revised)

    assert changes[0].type == "replace"
    assert changes[0].inline_diff == []


def test_build_section_diff_returns_empty_string_for_unchanged_text():
    assert build_section_diff_impl("etc_info", "같은 내용", "같은 내용") == ""
