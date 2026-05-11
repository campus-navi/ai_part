import pytest
from fastapi.testclient import TestClient

from app.analyzer import OpenAIAnalyzerConfig, OpenAINoticeAnalyzer
from app.main import app, get_notice_analyzer
from app.models import OfficialProcessRequest, OfficialProcessResponse


class FakeAnalyzer:
    def __init__(self) -> None:
        self.requests: list[OfficialProcessRequest] = []

    async def analyze(self, request: OfficialProcessRequest) -> OfficialProcessResponse:
        self.requests.append(request)
        if not request.structured_text.strip():
            raise ValueError("structured_text must not be blank")

        return OfficialProcessResponse(
            summary="2026학년도 2학기 교내장학금 신청 안내입니다.",
            target_grade_min=2,
            target_grade_max=4,
            tag_code="SCHOLARSHIP",
            keywords=["교내장학금", "성적우수장학금"],
            contact_phone="02-3290-1234",
            contact_email=None,
            start_date="2026-04-01",
            start_time="09:00:00",
            end_date="2026-05-31",
            end_time=None,
            required_documents="성적증명서, 재학증명서",
            apply_method_type="OTHER",
            apply_method_detail="장학 담당 부서 안내에 따라 제출",
            eligibility="현재 3학기 이상 이수 중인 재학생\n편입생은 한 학기 이상 이수 후 신청 가능",
            is_applicable=True,
        )


@pytest.fixture
def fake_analyzer() -> FakeAnalyzer:
    analyzer = FakeAnalyzer()
    app.dependency_overrides[get_notice_analyzer] = lambda: analyzer
    try:
        yield analyzer
    finally:
        app.dependency_overrides.clear()


def test_process_official_notice_returns_ai_response(fake_analyzer: FakeAnalyzer):
    client = TestClient(app)

    response = client.post(
        "/ai/official/process",
        json={
            "post_id": 123,
            "structured_text": "2026학년도 2학기 교내장학금 신청 안내...",
            "image_urls": ["https://cdn.example.com/notice.png"],
            "attachment_urls": ["https://s3.amazonaws.com/example/notice.pdf"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "summary": "2026학년도 2학기 교내장학금 신청 안내입니다.",
        "target_grade_min": 2,
        "target_grade_max": 4,
        "tag_code": "SCHOLARSHIP",
        "keywords": ["교내장학금", "성적우수장학금"],
        "contact_phone": "02-3290-1234",
        "contact_email": None,
        "start_date": "2026-04-01",
        "start_time": "09:00:00",
        "end_date": "2026-05-31",
        "end_time": None,
        "required_documents": "성적증명서, 재학증명서",
        "apply_method_type": "OTHER",
        "apply_method_detail": "장학 담당 부서 안내에 따라 제출",
        "eligibility": "현재 3학기 이상 이수 중인 재학생\n편입생은 한 학기 이상 이수 후 신청 가능",
        "is_applicable": True,
    }
    assert fake_analyzer.requests[0].image_urls == ["https://cdn.example.com/notice.png"]
    assert fake_analyzer.requests[0].attachment_urls == ["https://s3.amazonaws.com/example/notice.pdf"]


def test_batch_process_keeps_successes_when_one_item_fails(fake_analyzer: FakeAnalyzer):
    client = TestClient(app)

    response = client.post(
        "/ai/official/process/batch",
        json={
            "items": [
                {
                    "post_id": 125,
                    "structured_text": "인턴 채용공고입니다.",
                    "image_urls": [],
                    "attachment_urls": [],
                },
                {
                    "post_id": 126,
                    "structured_text": "   ",
                    "image_urls": [],
                    "attachment_urls": [],
                },
            ]
        },
    )

    assert response.status_code == 200
    results = {item["post_id"]: item for item in response.json()["results"]}

    assert results[125]["success"] is True
    assert results[125]["reason"] is None
    assert results[125]["result"]["tag_code"] == "SCHOLARSHIP"
    assert results[126]["success"] is False
    assert "structured_text" in results[126]["reason"]
    assert "AI 분석 오류" in results[126]["reason"]
    assert results[126]["result"] is None


def test_openai_analyzer_builds_multimodal_input():
    analyzer = OpenAINoticeAnalyzer(
        OpenAIAnalyzerConfig(max_images=1, max_attachments=1)
    )
    request = OfficialProcessRequest(
        post_id=123,
        structured_text="공지 본문",
        image_urls=[
            "https://cdn.example.com/notice-1.png",
            "https://cdn.example.com/notice-2.png",
        ],
        attachment_urls=[
            "https://cdn.example.com/files/notice%20guide.pdf",
            "https://cdn.example.com/files/extra.pdf",
        ],
    )

    payload = analyzer._build_input(request)

    content = payload[0]["content"]
    assert content[0] == {
        "type": "input_text",
        "text": "post_id: 123\nstructured_text:\n공지 본문",
    }
    assert content[1] == {
        "type": "input_image",
        "image_url": "https://cdn.example.com/notice-1.png",
        "detail": "auto",
    }
    assert content[2] == {
        "type": "input_file",
        "file_url": "https://cdn.example.com/files/notice%20guide.pdf",
        "filename": "notice guide.pdf",
    }
    assert len(content) == 3
