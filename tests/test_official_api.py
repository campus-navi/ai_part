import pytest
from fastapi.testclient import TestClient

from app.attachments import (
    ExtractedHwpAttachment,
    ExtractedPdfAttachment,
    HwpPreprocessResult,
    PdfPreprocessResult,
)
from app.analyzer import (
    OpenAIAnalyzerConfig,
    OpenAINoticeAnalyzer,
    _has_analysis_source,
    _normalize_response,
)
from app.main import app, get_notice_analyzer
from app.models import OfficialProcessRequest, OfficialProcessResponse


class FakeAnalyzer:
    def __init__(self) -> None:
        self.requests: list[OfficialProcessRequest] = []

    async def analyze(self, request: OfficialProcessRequest) -> OfficialProcessResponse:
        self.requests.append(request)
        if not _has_analysis_source(request):
            raise ValueError("structured_text, image_urls, or attachment_urls must include content")

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
def anyio_backend():
    return "asyncio"


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


def test_process_official_notice_accepts_null_structured_text_with_image(
    fake_analyzer: FakeAnalyzer,
):
    client = TestClient(app)

    response = client.post(
        "/ai/official/process",
        json={
            "post_id": 124,
            "structured_text": None,
            "image_urls": ["https://cdn.example.com/image-only-notice.png"],
            "attachment_urls": [],
        },
    )

    assert response.status_code == 200
    assert fake_analyzer.requests[0].structured_text is None
    assert fake_analyzer.requests[0].image_urls == ["https://cdn.example.com/image-only-notice.png"]


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


def test_normalize_response_clears_application_fields_when_not_applicable():
    response = OfficialProcessResponse(
        summary="구술면접 시험 안내입니다.",
        target_grade_min=None,
        target_grade_max=None,
        tag_code="ACADEMIC",
        keywords=["구술면접"],
        contact_phone="02-3290-5973",
        contact_email=None,
        start_date="2026-05-16",
        start_time="10:00:00",
        end_date="2026-05-16",
        end_time=None,
        required_documents="수험표, 신분증",
        apply_method_type="OTHER",
        apply_method_detail="입실 후 면접",
        eligibility="수험생",
        is_applicable=False,
    )

    normalized = _normalize_response(response)

    assert normalized.start_date is None
    assert normalized.start_time is None
    assert normalized.end_date is None
    assert normalized.end_time is None
    assert normalized.required_documents is None
    assert normalized.apply_method_type is None
    assert normalized.apply_method_detail is None
    assert normalized.eligibility is None


def test_normalize_response_clears_apply_method_detail_unless_other():
    response = OfficialProcessResponse(
        summary="장학금 신청 안내입니다.",
        target_grade_min=None,
        target_grade_max=None,
        tag_code="SCHOLARSHIP",
        keywords=["장학금"],
        contact_phone=None,
        contact_email=None,
        start_date="2026-05-06",
        start_time="10:00:00",
        end_date="2026-05-22",
        end_time="17:00:00",
        required_documents="신청서, 동의서",
        apply_method_type="FILE",
        apply_method_detail="화성시인재육성재단 홈페이지 내 온라인 신청",
        eligibility="학자금 지원구간 5구간 이내 대학생",
        is_applicable=True,
    )

    normalized = _normalize_response(response)

    assert normalized.apply_method_type == "FILE"
    assert normalized.apply_method_detail is None


def test_analysis_instructions_keep_required_documents_concise():
    from app.analyzer import ANALYSIS_INSTRUCTIONS

    assert "required_documents" in ANALYSIS_INSTRUCTIONS
    assert "comma-separated concise" in ANALYSIS_INSTRUCTIONS


class FakeAttachmentPreprocessor:
    async def preprocess(self, urls: list[str]) -> PdfPreprocessResult:
        if not urls:
            return PdfPreprocessResult()

        return PdfPreprocessResult(
            extracted=[
                ExtractedPdfAttachment(
                    url=urls[0],
                    filename="notice guide.pdf",
                    markdown="# PDF 안내\n장학금 신청 서류 안내",
                )
            ],
            fallback_urls=urls[1:],
        )

    def check_preflight(self):
        return type("Preflight", (), {"warnings": []})()


class FallbackOnlyAttachmentPreprocessor:
    async def preprocess(self, urls: list[str]) -> PdfPreprocessResult:
        return PdfPreprocessResult(fallback_urls=urls)

    def check_preflight(self):
        return type("Preflight", (), {"warnings": []})()


class FakeHwpAttachmentPreprocessor:
    async def preprocess(self, urls: list[str]) -> HwpPreprocessResult:
        extracted = [
            ExtractedHwpAttachment(
                url=url,
                filename=url.rsplit("/", 1)[-1].split("?", 1)[0],
                text="교내장학금 신청서와 개인정보 동의서를 제출해야 합니다.",
            )
            for url in urls
            if url.split("?", 1)[0].lower().endswith((".hwp", ".hwpx"))
        ]
        extracted_urls = {item.url for item in extracted}
        return HwpPreprocessResult(
            extracted=extracted,
            fallback_urls=[url for url in urls if url not in extracted_urls],
        )

    def check_preflight(self):
        return type("Preflight", (), {"warnings": []})()


class NoopHwpAttachmentPreprocessor:
    async def preprocess(self, urls: list[str]) -> HwpPreprocessResult:
        return HwpPreprocessResult(fallback_urls=urls)

    def check_preflight(self):
        return type("Preflight", (), {"warnings": []})()


@pytest.mark.anyio
async def test_openai_analyzer_builds_multimodal_input():
    analyzer = OpenAINoticeAnalyzer(
        OpenAIAnalyzerConfig(max_images=1, max_attachments=2),
        attachment_preprocessor=FakeAttachmentPreprocessor(),
        hwp_attachment_preprocessor=NoopHwpAttachmentPreprocessor(),
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

    payload = await analyzer._build_input(request)

    content = payload[0]["content"]
    assert content[0]["type"] == "input_text"
    assert content[0]["text"].startswith("post_id: 123\nstructured_text:\n공지 본문")
    assert "PDF attachments extracted by OpenDataLoader" in content[0]["text"]
    assert "untrusted attachment content" in content[0]["text"]
    assert "# PDF 안내\n장학금 신청 서류 안내" in content[0]["text"]
    assert content[1] == {
        "type": "input_image",
        "image_url": "https://cdn.example.com/notice-1.png",
        "detail": "auto",
    }
    assert content[2] == {
        "type": "input_file",
        "file_url": "https://cdn.example.com/files/extra.pdf",
    }
    assert len(content) == 3


@pytest.mark.anyio
async def test_openai_analyzer_builds_multimodal_input_with_null_structured_text():
    analyzer = OpenAINoticeAnalyzer(
        OpenAIAnalyzerConfig(max_images=1, max_attachments=2),
        attachment_preprocessor=FakeAttachmentPreprocessor(),
        hwp_attachment_preprocessor=NoopHwpAttachmentPreprocessor(),
    )
    request = OfficialProcessRequest(
        post_id=124,
        structured_text=None,
        image_urls=["https://cdn.example.com/image-only-notice.png"],
        attachment_urls=[],
    )

    payload = await analyzer._build_input(request)

    content = payload[0]["content"]
    assert content[0]["type"] == "input_text"
    assert content[0]["text"] == "post_id: 124\nstructured_text:\n"
    assert content[1] == {
        "type": "input_image",
        "image_url": "https://cdn.example.com/image-only-notice.png",
        "detail": "auto",
    }


@pytest.mark.anyio
async def test_openai_analyzer_filters_fallback_attachments_for_openai_support():
    analyzer = OpenAINoticeAnalyzer(
        OpenAIAnalyzerConfig(max_images=2, max_attachments=8),
        attachment_preprocessor=FallbackOnlyAttachmentPreprocessor(),
        hwp_attachment_preprocessor=NoopHwpAttachmentPreprocessor(),
    )
    request = OfficialProcessRequest(
        post_id=3,
        structured_text="모집기간 :2026. 5. 6.(수) 10시 ~ 5. 22.(금) 17시",
        image_urls=["https://cdn.example.com/images/notice.png?X-Amz-Signature=abc123"],
        attachment_urls=[
            "https://cdn.example.com/files/guide.hwp?X-Amz-Signature=def456",
            "https://cdn.example.com/files/fallback.pdf?X-Amz-Signature=ghi789",
            "https://cdn.example.com/files/archive.zip?X-Amz-Signature=jkl012",
            "https://cdn.example.com/files/form.docx?X-Amz-Signature=pqr678",
            "https://cdn.example.com/files/sheet.xlsx?X-Amz-Signature=stu901",
            "https://cdn.example.com/files/poster.webp?X-Amz-Signature=mno345",
            "https://cdn.example.com/files/slides.pptx?X-Amz-Signature=vwx234",
            "https://cdn.example.com/files/readme.md?X-Amz-Signature=yz567",
        ],
    )

    payload = await analyzer._build_input(request)

    content = payload[0]["content"]
    assert {"type": "input_file", "file_url": request.attachment_urls[1]} in content
    assert {"type": "input_file", "file_url": request.attachment_urls[3]} in content
    assert {"type": "input_file", "file_url": request.attachment_urls[4]} in content
    assert {"type": "input_file", "file_url": request.attachment_urls[6]} in content
    assert {"type": "input_file", "file_url": request.attachment_urls[7]} in content
    assert all(item.get("filename") is None for item in content)
    assert {"type": "input_image", "image_url": request.attachment_urls[5], "detail": "auto"} in content
    assert not any("guide.hwp" in str(item) for item in content[1:])
    assert not any("archive.zip" in str(item) for item in content[1:])
    assert "Skipped unsupported attachments" in content[0]["text"]
    assert "guide.hwp" in content[0]["text"]
    assert "archive.zip" in content[0]["text"]
    assert "form.docx" not in content[0]["text"]
    assert "sheet.xlsx" not in content[0]["text"]
    assert "slides.pptx" not in content[0]["text"]
    assert "readme.md" not in content[0]["text"]


@pytest.mark.anyio
async def test_openai_analyzer_uses_content_disposition_filename_for_file_support():
    analyzer = OpenAINoticeAnalyzer(
        OpenAIAnalyzerConfig(max_images=2, max_attachments=3),
        attachment_preprocessor=FallbackOnlyAttachmentPreprocessor(),
        hwp_attachment_preprocessor=NoopHwpAttachmentPreprocessor(),
    )
    request = OfficialProcessRequest(
        post_id=4,
        structured_text="첨부파일을 확인하세요.",
        attachment_urls=[
            "https://cdn.example.com/files/notice.pdf",
            (
                "https://cdn.example.com/files/download.pdf?"
                "response-content-disposition=attachment%3B%20filename%3Dnotice.docx"
            ),
        ],
    )

    payload = await analyzer._build_input(request)

    content = payload[0]["content"]
    assert {"type": "input_file", "file_url": request.attachment_urls[0]} in content
    assert {"type": "input_file", "file_url": request.attachment_urls[1]} in content
    assert "Skipped unsupported attachments" not in content[0]["text"]


@pytest.mark.anyio
async def test_openai_analyzer_includes_extracted_hwp_and_hwpx_text():
    analyzer = OpenAINoticeAnalyzer(
        OpenAIAnalyzerConfig(max_images=2, max_attachments=4),
        attachment_preprocessor=FallbackOnlyAttachmentPreprocessor(),
        hwp_attachment_preprocessor=FakeHwpAttachmentPreprocessor(),
    )
    request = OfficialProcessRequest(
        post_id=5,
        structured_text="첨부 양식을 확인하세요.",
        attachment_urls=[
            "https://cdn.example.com/files/guide.hwp?X-Amz-Signature=abc123",
            "https://cdn.example.com/files/form.hwpx?X-Amz-Signature=def456",
            "https://cdn.example.com/files/archive.zip?X-Amz-Signature=ghi789",
        ],
    )

    payload = await analyzer._build_input(request)

    content = payload[0]["content"]
    assert not any("guide.hwp" in str(item) for item in content[1:])
    assert not any("form.hwpx" in str(item) for item in content[1:])
    assert "HWP/HWPX attachments extracted by rhwp" in content[0]["text"]
    assert "교내장학금 신청서와 개인정보 동의서를 제출해야 합니다." in content[0]["text"]
    assert "Skipped unsupported attachments" in content[0]["text"]
    assert "archive.zip" in content[0]["text"]
    assert "guide.hwp" not in content[0]["text"].split("Skipped unsupported attachments", 1)[1]
    assert "form.hwpx" not in content[0]["text"].split("Skipped unsupported attachments", 1)[1]
