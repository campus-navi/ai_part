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
            max_tokens=1200,
        )
    )
    request = AcademicPlanReviewRequest(
        document_type="ACADEMIC_PLAN",
        metadata={
            "major_type": "DOUBLE_MAJOR",
            "target_name": "경제학과",
            "user_department": "컴퓨터공학과",
        },
        sections=[
            {
                "section_key": "application_motive",
                "content": "경제 현상에 관심이 있어 경제학을 더 배우고 싶습니다.",
            }
        ],
    )

    response = await reviewer.review(request)
    section = response.sections[0]
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
    assert section.section_key == "application_motive"
    assert section.original_content == request.sections[0].content
    assert section.revised_content
    assert section.revised_content != section.original_content
    assert "--- application_motive.original" in section.diff
    assert section.changes
    assert section.reasons
