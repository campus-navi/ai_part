from fastapi.testclient import TestClient

from app.academic_plan import (
    AcademicPlanMetadata,
    AcademicPlanReviewRequest,
    AcademicPlanReviewResponse,
    AcademicPlanSectionReview,
    RevisionReason,
    _load_academic_plan_skill_impl,
    _normalize_review_response,
    _retrieve_reference_examples_impl,
    build_section_diff_impl,
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
    examples = _retrieve_reference_examples_impl("경제학과", "컴퓨터공학과", "study_plan")

    assert skill["major_type_label"] == "이중전공"
    assert "학업계획" in skill["section_title"]
    assert examples[0]["source_type"] == "example"
    assert "과목" in examples[0]["content"]


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


def test_build_section_diff_returns_empty_string_for_unchanged_text():
    assert build_section_diff_impl("etc_info", "같은 내용", "같은 내용") == ""
