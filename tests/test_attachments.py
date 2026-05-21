from pathlib import Path

import httpx
import pytest

from app.attachments import (
    DownloadedPdf,
    DownloadedHwpAttachment,
    ExtractedHwpAttachment,
    HwpAttachmentPreprocessor,
    HwpExtractError,
    HwpPreprocessConfig,
    HttpxPdfDownloader,
    OpenDataLoaderConverter,
    PdfAttachmentPreprocessor,
    PdfConverterError,
    PdfDownloadError,
    PdfPreprocessConfig,
    PreflightResult,
    UnsupportedAttachmentError,
    safe_filename_from_url,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class PassingPreflight:
    def check(self) -> PreflightResult:
        return PreflightResult(available=True)


class FailingPreflight:
    def check(self) -> PreflightResult:
        return PreflightResult(available=False, warnings=["Java 11+ runtime is unavailable"])


class FakeDownloader:
    async def download_pdf(
        self,
        url: str,
        destination: Path,
        *,
        max_bytes: int,
        timeout_seconds: float,
    ) -> DownloadedPdf:
        filename = safe_filename_from_url(url, default="attachment.pdf")
        path = destination / filename
        path.write_bytes(b"%PDF-1.7")
        return DownloadedPdf(url=url, path=path, filename=filename)


class SkippingDownloader:
    async def download_pdf(
        self,
        url: str,
        destination: Path,
        *,
        max_bytes: int,
        timeout_seconds: float,
    ) -> DownloadedPdf:
        raise UnsupportedAttachmentError("attachment is not a PDF")


class WritingConverter:
    def convert(
        self,
        input_paths: list[Path],
        output_dir: Path,
        config: PdfPreprocessConfig,
    ) -> None:
        for input_path in input_paths:
            (output_dir / f"{input_path.stem}.md").write_text(
                "# PDF 안내\n장학금 신청 서류 안내",
                encoding="utf-8",
            )
            (output_dir / f"{input_path.stem}.json").write_text(
                '{"pages": []}',
                encoding="utf-8",
            )


class FailingConverter:
    def convert(
        self,
        input_paths: list[Path],
        output_dir: Path,
        config: PdfPreprocessConfig,
    ) -> None:
        raise PdfConverterError("hybrid backend unavailable")


class SlowConverter:
    def convert(
        self,
        input_paths: list[Path],
        output_dir: Path,
        config: PdfPreprocessConfig,
    ) -> None:
        import time

        time.sleep(0.05)


class FakeHwpDownloader:
    async def download_hwp(
        self,
        url: str,
        destination: Path,
        *,
        max_bytes: int,
        timeout_seconds: float,
    ) -> DownloadedHwpAttachment:
        filename = safe_filename_from_url(url, default="attachment.hwp")
        path = destination / filename
        path.write_bytes(b"hwp bytes")
        return DownloadedHwpAttachment(url=url, path=path, filename=filename)


class SkippingHwpDownloader:
    async def download_hwp(
        self,
        url: str,
        destination: Path,
        *,
        max_bytes: int,
        timeout_seconds: float,
    ) -> DownloadedHwpAttachment:
        raise UnsupportedAttachmentError("attachment URL extension is not HWP/HWPX")


class FakeHwpExtractor:
    def extract(self, path: Path) -> str:
        return f"{path.name} 신청서 제출 안내"


class FailingHwpExtractor:
    def extract(self, path: Path) -> str:
        raise HwpExtractError("rhwp parse failed")


def test_safe_filename_from_url_uses_ascii_only_for_java_compatibility():
    filename = safe_filename_from_url(
        "https://cdn.example.com/files/2026%ED%95%99%EB%85%84%EB%8F%84%20%ED%9B%84%EA%B8%B0%20%EC%9E%85%EC%8B%9C%20%EC%95%88%EB%82%B4%EB%AC%B8.pdf"
    )

    assert filename == "2026.pdf"


@pytest.mark.anyio
async def test_preprocess_extracts_pdf_markdown_and_omits_fallback_url():
    preprocessor = PdfAttachmentPreprocessor(
        config=PdfPreprocessConfig(),
        downloader=FakeDownloader(),
        converter=WritingConverter(),
        preflight=PassingPreflight(),
    )

    result = await preprocessor.preprocess(["https://cdn.example.com/files/notice.pdf"])

    assert result.fallback_urls == []
    assert result.warnings == []
    assert len(result.extracted) == 1
    assert result.extracted[0].url == "https://cdn.example.com/files/notice.pdf"
    assert result.extracted[0].filename == "notice.pdf"
    assert "# PDF 안내" in result.extracted[0].markdown


@pytest.mark.anyio
async def test_hwp_preprocess_extracts_text_and_omits_fallback_url():
    preprocessor = HwpAttachmentPreprocessor(
        config=HwpPreprocessConfig(),
        downloader=FakeHwpDownloader(),
        extractor=FakeHwpExtractor(),
        preflight=PassingPreflight(),
    )

    result = await preprocessor.preprocess(
        [
            "https://cdn.example.com/files/notice.hwp",
            "https://cdn.example.com/files/form.hwpx",
        ]
    )

    assert result.fallback_urls == []
    assert result.warnings == []
    assert result.extracted == [
        ExtractedHwpAttachment(
            url="https://cdn.example.com/files/notice.hwp",
            filename="notice.hwp",
            text="notice.hwp 신청서 제출 안내",
        ),
        ExtractedHwpAttachment(
            url="https://cdn.example.com/files/form.hwpx",
            filename="form.hwpx",
            text="form.hwpx 신청서 제출 안내",
        ),
    ]


@pytest.mark.anyio
async def test_hwp_preprocess_falls_back_for_unsupported_attachment():
    preprocessor = HwpAttachmentPreprocessor(
        config=HwpPreprocessConfig(),
        downloader=SkippingHwpDownloader(),
        extractor=FakeHwpExtractor(),
        preflight=PassingPreflight(),
    )

    result = await preprocessor.preprocess(["https://cdn.example.com/files/archive.zip"])

    assert result.extracted == []
    assert result.fallback_urls == ["https://cdn.example.com/files/archive.zip"]
    assert "extension is not HWP/HWPX" in result.warnings[0]


@pytest.mark.anyio
async def test_hwp_preprocess_falls_back_when_extraction_fails():
    preprocessor = HwpAttachmentPreprocessor(
        config=HwpPreprocessConfig(),
        downloader=FakeHwpDownloader(),
        extractor=FailingHwpExtractor(),
        preflight=PassingPreflight(),
    )

    result = await preprocessor.preprocess(["https://cdn.example.com/files/notice.hwp"])

    assert result.extracted == []
    assert result.fallback_urls == ["https://cdn.example.com/files/notice.hwp"]
    assert "rhwp parse failed" in result.warnings[0]


@pytest.mark.anyio
async def test_hwp_preprocess_truncates_extracted_text():
    preprocessor = HwpAttachmentPreprocessor(
        config=HwpPreprocessConfig(extracted_text_max_chars=10),
        downloader=FakeHwpDownloader(),
        extractor=FakeHwpExtractor(),
        preflight=PassingPreflight(),
    )

    result = await preprocessor.preprocess(["https://cdn.example.com/files/notice.hwp"])

    assert result.extracted[0].text == "notice.hwp\n\n[HWP/HWPX text truncated]"


@pytest.mark.anyio
async def test_preprocess_falls_back_to_url_when_converter_fails():
    preprocessor = PdfAttachmentPreprocessor(
        config=PdfPreprocessConfig(),
        downloader=FakeDownloader(),
        converter=FailingConverter(),
        preflight=PassingPreflight(),
    )

    result = await preprocessor.preprocess(["https://cdn.example.com/files/notice.pdf"])

    assert result.extracted == []
    assert result.fallback_urls == ["https://cdn.example.com/files/notice.pdf"]
    assert "hybrid backend unavailable" in result.warnings[0]


@pytest.mark.anyio
async def test_preprocess_falls_back_when_preflight_fails():
    preprocessor = PdfAttachmentPreprocessor(
        config=PdfPreprocessConfig(),
        downloader=FakeDownloader(),
        converter=WritingConverter(),
        preflight=FailingPreflight(),
    )

    result = await preprocessor.preprocess(["https://cdn.example.com/files/notice.pdf"])

    assert result.extracted == []
    assert result.fallback_urls == ["https://cdn.example.com/files/notice.pdf"]
    assert result.warnings == ["Java 11+ runtime is unavailable"]


@pytest.mark.anyio
async def test_preprocess_falls_back_when_converter_times_out():
    preprocessor = PdfAttachmentPreprocessor(
        config=PdfPreprocessConfig(convert_timeout_seconds=0.001),
        downloader=FakeDownloader(),
        converter=SlowConverter(),
        preflight=PassingPreflight(),
    )

    result = await preprocessor.preprocess(["https://cdn.example.com/files/notice.pdf"])

    assert result.extracted == []
    assert result.fallback_urls == ["https://cdn.example.com/files/notice.pdf"]
    assert result.warnings == ["OpenDataLoader conversion timed out"]


@pytest.mark.anyio
async def test_preprocess_falls_back_for_non_pdf_attachment():
    preprocessor = PdfAttachmentPreprocessor(
        config=PdfPreprocessConfig(),
        downloader=SkippingDownloader(),
        converter=WritingConverter(),
        preflight=PassingPreflight(),
    )

    result = await preprocessor.preprocess(["https://cdn.example.com/files/notice.docx"])

    assert result.extracted == []
    assert result.fallback_urls == ["https://cdn.example.com/files/notice.docx"]
    assert "attachment is not a PDF" in result.warnings[0]


@pytest.mark.anyio
async def test_preprocess_tracks_mislabelled_pdf_url_as_unsupported():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": (
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                )
            },
            content=b"not a pdf",
            request=request,
        )

    preprocessor = PdfAttachmentPreprocessor(
        config=PdfPreprocessConfig(),
        downloader=HttpxPdfDownloader(transport=httpx.MockTransport(handler)),
        converter=WritingConverter(),
        preflight=PassingPreflight(),
    )

    result = await preprocessor.preprocess(["https://cdn.example.com/files/notice.pdf"])

    assert result.extracted == []
    assert result.fallback_urls == ["https://cdn.example.com/files/notice.pdf"]
    assert "attachment is not a PDF" in result.warnings[0]


@pytest.mark.anyio
async def test_httpx_downloader_rejects_non_pdf_content_type(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            content=b"not a pdf",
            request=request,
        )

    downloader = HttpxPdfDownloader(transport=httpx.MockTransport(handler))

    with pytest.raises(UnsupportedAttachmentError, match="attachment is not a PDF"):
        await downloader.download_pdf(
            "https://cdn.example.com/files/notice",
            tmp_path,
            max_bytes=1024,
            timeout_seconds=1,
        )


@pytest.mark.anyio
async def test_httpx_downloader_enforces_size_limit(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "application/pdf",
                "content-length": "2048",
            },
            content=b"",
            request=request,
        )

    downloader = HttpxPdfDownloader(transport=httpx.MockTransport(handler))

    with pytest.raises(PdfDownloadError, match="size limit"):
        await downloader.download_pdf(
            "https://cdn.example.com/files/notice.pdf",
            tmp_path,
            max_bytes=1024,
            timeout_seconds=1,
        )


@pytest.mark.anyio
async def test_httpx_downloader_wraps_timeout(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("boom", request=request)

    downloader = HttpxPdfDownloader(transport=httpx.MockTransport(handler))

    with pytest.raises(PdfDownloadError, match="timed out"):
        await downloader.download_pdf(
            "https://cdn.example.com/files/notice.pdf",
            tmp_path,
            max_bytes=1024,
            timeout_seconds=1,
        )


@pytest.mark.anyio
async def test_httpx_downloader_skips_clear_non_pdf_extension_without_network(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("non-PDF attachments should not be downloaded")

    downloader = HttpxPdfDownloader(transport=httpx.MockTransport(handler))

    with pytest.raises(UnsupportedAttachmentError, match="extension is not PDF"):
        await downloader.download_pdf(
            "https://cdn.example.com/files/notice.docx",
            tmp_path,
            max_bytes=1024,
            timeout_seconds=1,
        )


def test_opendataloader_converter_passes_runtime_options(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class FakeOpenDataLoader:
        @staticmethod
        def convert(**kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "opendataloader_pdf", FakeOpenDataLoader)

    converter = OpenDataLoaderConverter()
    input_pdf = tmp_path / "notice.pdf"
    input_pdf.write_bytes(b"%PDF-1.7")

    converter.convert(
        [input_pdf],
        tmp_path / "out",
        PdfPreprocessConfig(
            hybrid="",
            hybrid_mode="full",
            hybrid_timeout_ms=120_000,
            quiet=False,
            hybrid_fallback=True,
        ),
    )

    assert captured["hybrid"] is None
    assert captured["hybrid_mode"] is None
    assert captured["hybrid_timeout"] is None
    assert captured["quiet"] is False
    assert captured["hybrid_fallback"] is True


def test_opendataloader_converter_retries_java_only_when_hybrid_fails(
    monkeypatch,
    tmp_path,
):
    calls: list[dict[str, object]] = []

    class FakeOpenDataLoader:
        @staticmethod
        def convert(**kwargs):
            calls.append(kwargs)
            if kwargs["hybrid"] == "docling-fast":
                raise RuntimeError("Hybrid server is not available")

    monkeypatch.setitem(__import__("sys").modules, "opendataloader_pdf", FakeOpenDataLoader)

    converter = OpenDataLoaderConverter()
    input_pdf = tmp_path / "notice.pdf"
    input_pdf.write_bytes(b"%PDF-1.7")

    converter.convert(
        [input_pdf],
        tmp_path / "out",
        PdfPreprocessConfig(
            hybrid="docling-fast",
            hybrid_mode="full",
            quiet=False,
            hybrid_fallback=True,
        ),
    )

    assert len(calls) == 2
    assert calls[0]["hybrid"] == "docling-fast"
    assert calls[1]["hybrid"] is None
    assert calls[1]["hybrid_mode"] is None
