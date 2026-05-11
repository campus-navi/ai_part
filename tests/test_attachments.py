from pathlib import Path

import httpx
import pytest

from app.attachments import (
    DownloadedPdf,
    HttpxPdfDownloader,
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
            "https://cdn.example.com/files/notice.txt",
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
