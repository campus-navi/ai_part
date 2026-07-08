import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from app.academic_plan import (
    AcademicPlanAgentConfig,
    AcademicPlanReviewRequest,
    OpenAIAcademicPlanReviewer,
)


pytestmark = pytest.mark.integration


def _apply_suggestions(text, suggestions):
    result = text
    for suggestion in sorted(suggestions, key=lambda item: item.original_range.start, reverse=True):
        result = result[: suggestion.original_range.start] + (suggestion.suggested_text or "") + result[
            suggestion.original_range.end :
        ]
    return result


def _repeat_to_length(text: str, min_length: int = 720, max_length: int = 900) -> str:
    chunks: list[str] = []
    while len("\n\n".join(chunks)) < min_length:
        chunks.append(text.strip())
    return "\n\n".join(chunks)[:max_length]


def _long_academic_plan_request() -> AcademicPlanReviewRequest:
    sections = [
        {
            "section_key": "application_motive",
            "content": _repeat_to_length(
                "컴퓨터공학 전공에서 자료구조와 데이터 분석을 배우며 기술이 사회 현상을 설명하는 데 유용하다는 점을 체감했습니다. "
                "하지만 서비스 이용자의 선택, 가격 변화, 플랫폼 정책의 효과를 해석하려면 경제학의 수요와 공급, 계량 분석, 산업조직론 지식이 필요했습니다. "
                "경제학과 이중전공을 통해 기존 전공의 구현 역량을 경제 현상의 원인과 결과를 분석하는 학문적 틀로 확장하고 싶습니다."
            ),
        },
        {
            "section_key": "interest_field",
            "content": _repeat_to_length(
                "관심 분야는 플랫폼 시장과 데이터 기반 의사결정입니다. 컴퓨터공학 수업에서 추천 시스템과 사용자 로그 분석을 접하면서 알고리즘의 성능만으로는 시장의 변화를 설명하기 어렵다는 점을 느꼈습니다. "
                "이후 관련 교양과 공개 강좌를 통해 소비자 선택, 네트워크 효과, 정보 비대칭 개념을 찾아보며 기술적 문제와 경제학적 문제가 연결된다는 점을 확인했습니다. "
                "앞으로는 데이터 처리 역량을 바탕으로 시장 참여자의 행동을 정량적으로 해석하는 방향을 탐구하고 싶습니다."
            ),
        },
        {
            "section_key": "study_plan",
            "content": _repeat_to_length(
                "학업 계획은 기초 이론을 먼저 보완한 뒤 계량 분석과 응용 분야로 확장하는 순서로 세우고 있습니다. 3학년에는 미시경제학, 거시경제학, 경제통계학을 우선 수강해 경제학의 기본 언어와 분석 단위를 익히겠습니다. "
                "동시에 컴퓨터공학에서 배운 Python과 데이터베이스 활용 능력을 계량경제학 수업의 실증 분석 과제에 연결하겠습니다. "
                "4학년에는 산업조직론과 데이터 관련 프로젝트 과목을 수강하며 플랫폼 시장 사례를 분석하고, 졸업 전에는 분석 결과를 보고서 형태로 정리하겠습니다."
            ),
        },
        {
            "section_key": "etc_info",
            "content": _repeat_to_length(
                "기타 항목에서는 경제학과 진입을 준비하며 쌓은 실천적 근거를 보완하고자 합니다. 컴퓨터공학 전공 과제에서 팀 프로젝트를 진행할 때 기능 구현뿐 아니라 사용자의 선택 기준과 서비스 지속 가능성을 함께 검토했습니다. "
                "또한 통계와 데이터 분석 관련 학습을 병행하며 단순한 관심이 아니라 실제 분석을 수행할 수 있는 기반을 마련했습니다. "
                "지원 후에는 기존 전공의 프로그래밍 역량을 경제학 학습의 보조 도구로 활용해 수업 과제와 프로젝트에서 검증 가능한 산출물을 만들겠습니다."
            ),
        },
    ]
    return AcademicPlanReviewRequest(
        document_type="ACADEMIC_PLAN",
        metadata={
            "major_type": "DOUBLE_MAJOR",
            "target_name": "경제학과",
            "user_department": "컴퓨터공학과",
        },
        sections=sections,
    )


def test_openai_api_case_uses_700_to_1000_character_sections():
    request = _long_academic_plan_request()

    assert {section.section_key for section in request.sections} == {
        "application_motive",
        "interest_field",
        "study_plan",
        "etc_info",
    }
    assert all(700 <= len(section.content) <= 1000 for section in request.sections)


@pytest.mark.anyio
async def test_academic_plan_review_calls_openai_api():
    if os.getenv("RUN_OPENAI_API_TESTS") != "1":
        pytest.skip("set RUN_OPENAI_API_TESTS=1 to call the OpenAI API")

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required")

    reviewer = OpenAIAcademicPlanReviewer(
        AcademicPlanAgentConfig(
            model=os.getenv("OPENAI_API_TEST_MODEL", "gpt-4.1-mini"),
            max_tokens=int(os.getenv("OPENAI_API_TEST_MAX_TOKENS", "4000")),
        )
    )
    request = _long_academic_plan_request()

    response = await reviewer.review(request)
    response_json = response.model_dump(mode="json")
    artifact_path = Path(__file__).resolve().parent / "artifacts" / "academic_plan_openai_latest.json"
    artifact_path.parent.mkdir(exist_ok=True)
    artifact_path.write_text(
        json.dumps(response_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nOpenAI academic plan review response:")
    print(json.dumps(response_json, ensure_ascii=False, indent=2))
    print(f"Saved response artifact: {artifact_path}")

    assert response.document_type == "ACADEMIC_PLAN"
    assert response.metadata.target_name == "경제학과"
    assert len(response.sections) == 4
    for original, reviewed in zip(request.sections, response.sections, strict=True):
        assert 700 <= len(original.content) <= 1000
        assert reviewed.section_key == original.section_key
        assert reviewed.original_content == original.content
        assert len(reviewed.revised_content) >= 650
        assert len(reviewed.revised_content) >= int(len(original.content) * 0.8)
        assert reviewed.revised_content
        assert reviewed.revised_content != reviewed.original_content
        assert f"--- {original.section_key}.original" in reviewed.diff
        assert reviewed.suggestions
        assert _apply_suggestions(reviewed.original_content, reviewed.suggestions) == reviewed.revised_content
        assert all(suggestion.reason.summary for suggestion in reviewed.suggestions)
        assert reviewed.reasons
        assert "revision_stages" not in reviewed.model_dump()
