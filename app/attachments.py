from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

import httpx


@dataclass(frozen=True)
class PdfPreprocessConfig:
    enabled: bool = True
    download_timeout_seconds: float = 10
    max_bytes: int = 10_485_760
    max_files: int = 8
    extracted_text_max_chars: int = 60_000
    convert_timeout_seconds: float = 180
    hybrid: str | None = "docling-fast"
    hybrid_mode: str = "full"
    hybrid_timeout_ms: int = 120_000
    quiet: bool = False
    hybrid_fallback: bool = True

    @classmethod
    def from_env(cls) -> PdfPreprocessConfig:
        return cls(
            enabled=_env_bool("PDF_PREPROCESS_ENABLED", cls.enabled),
            download_timeout_seconds=_env_float(
                "PDF_DOWNLOAD_TIMEOUT_SECONDS",
                cls.download_timeout_seconds,
            ),
            max_bytes=_env_int("PDF_MAX_BYTES", cls.max_bytes),
            max_files=_env_int("PDF_MAX_FILES", cls.max_files),
            extracted_text_max_chars=_env_int(
                "PDF_EXTRACTED_TEXT_MAX_CHARS",
                cls.extracted_text_max_chars,
            ),
            convert_timeout_seconds=_env_float(
                "PDF_CONVERT_TIMEOUT_SECONDS",
                cls.convert_timeout_seconds,
            ),
            hybrid=_env_hybrid("OPENDATALOADER_HYBRID", cls.hybrid),
            hybrid_mode=os.getenv("OPENDATALOADER_HYBRID_MODE", cls.hybrid_mode),
            hybrid_timeout_ms=_env_int(
                "OPENDATALOADER_HYBRID_TIMEOUT_MS",
                cls.hybrid_timeout_ms,
            ),
            quiet=_env_bool("OPENDATALOADER_QUIET", cls.quiet),
            hybrid_fallback=_env_bool(
                "OPENDATALOADER_HYBRID_FALLBACK",
                cls.hybrid_fallback,
            ),
        )


@dataclass(frozen=True)
class DownloadedPdf:
    url: str
    path: Path
    filename: str


@dataclass(frozen=True)
class ExtractedPdfAttachment:
    url: str
    filename: str
    markdown: str


@dataclass(frozen=True)
class PdfPreprocessResult:
    extracted: list[ExtractedPdfAttachment] = field(default_factory=list)
    fallback_urls: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PreflightResult:
    available: bool
    warnings: list[str] = field(default_factory=list)


class PdfDownloadError(RuntimeError):
    pass


class UnsupportedAttachmentError(RuntimeError):
    pass


class PdfConverterError(RuntimeError):
    pass


class PdfDownloader(Protocol):
    async def download_pdf(
        self,
        url: str,
        destination: Path,
        *,
        max_bytes: int,
        timeout_seconds: float,
    ) -> DownloadedPdf:
        ...


class PdfConverter(Protocol):
    def convert(
        self,
        input_paths: list[Path],
        output_dir: Path,
        config: PdfPreprocessConfig,
    ) -> None:
        ...


class PreflightChecker(Protocol):
    def check(self) -> PreflightResult:
        ...


class HttpxPdfDownloader:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.transport = transport

    async def download_pdf(
        self,
        url: str,
        destination: Path,
        *,
        max_bytes: int,
        timeout_seconds: float,
    ) -> DownloadedPdf:
        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise UnsupportedAttachmentError("only https PDF URLs are preprocessed")
        if _has_clear_non_pdf_extension(url):
            raise UnsupportedAttachmentError("attachment URL extension is not PDF")

        filename = safe_filename_from_url(url, default="attachment.pdf")
        if not filename.lower().endswith(".pdf"):
            filename = f"{filename}.pdf"
        output_path = _unique_path(destination / filename)

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=timeout_seconds,
                transport=self.transport,
            ) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    has_pdf_extension = urlparse(str(response.url)).path.lower().endswith(".pdf")
                    is_pdf = content_type == "application/pdf" or has_pdf_extension
                    if not is_pdf:
                        raise UnsupportedAttachmentError("attachment is not a PDF")

                    content_length = response.headers.get("content-length")
                    if content_length is not None and int(content_length) > max_bytes:
                        raise PdfDownloadError("PDF exceeds configured size limit")

                    total = 0
                    with output_path.open("wb") as file:
                        async for chunk in response.aiter_bytes():
                            total += len(chunk)
                            if total > max_bytes:
                                raise PdfDownloadError("PDF exceeds configured size limit")
                            file.write(chunk)
        except UnsupportedAttachmentError:
            raise
        except PdfDownloadError:
            raise
        except httpx.TimeoutException as exc:
            raise PdfDownloadError(f"PDF download timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise PdfDownloadError(f"PDF download failed: {exc}") from exc
        except ValueError as exc:
            raise PdfDownloadError(f"invalid PDF response headers: {exc}") from exc

        return DownloadedPdf(url=url, path=output_path, filename=output_path.name)


class OpenDataLoaderConverter:
    def convert(
        self,
        input_paths: list[Path],
        output_dir: Path,
        config: PdfPreprocessConfig,
    ) -> None:
        try:
            import opendataloader_pdf

            hybrid = _normalize_hybrid(config.hybrid)
            self._convert_with_options(
                opendataloader_pdf,
                input_paths,
                output_dir,
                config,
                hybrid=hybrid,
            )
        except Exception as exc:
            if _normalize_hybrid(config.hybrid) and config.hybrid_fallback:
                try:
                    import opendataloader_pdf

                    self._convert_with_options(
                        opendataloader_pdf,
                        input_paths,
                        output_dir,
                        config,
                        hybrid=None,
                    )
                    return
                except Exception as fallback_exc:
                    raise PdfConverterError(
                        f"OpenDataLoader conversion failed: {exc}; "
                        f"Java fallback failed: {fallback_exc}"
                    ) from fallback_exc

            raise PdfConverterError(f"OpenDataLoader conversion failed: {exc}") from exc

    def _convert_with_options(
        self,
        opendataloader_pdf,
        input_paths: list[Path],
        output_dir: Path,
        config: PdfPreprocessConfig,
        *,
        hybrid: str | None,
    ) -> None:
        opendataloader_pdf.convert(
            input_path=[str(path) for path in input_paths],
            output_dir=str(output_dir),
            format="markdown,json",
            hybrid=hybrid,
            hybrid_mode=config.hybrid_mode if hybrid else None,
            hybrid_timeout=str(config.hybrid_timeout_ms) if hybrid else None,
            hybrid_fallback=config.hybrid_fallback,
            quiet=config.quiet,
        )


class OpenDataLoaderPreflight:
    def __init__(self) -> None:
        self._cached: PreflightResult | None = None

    def check(self) -> PreflightResult:
        if self._cached is not None:
            return self._cached

        warnings: list[str] = []
        try:
            subprocess.run(
                ["java", "-version"],
                capture_output=True,
                check=True,
                timeout=5,
            )
        except Exception as exc:
            warnings.append(f"Java 11+ runtime is unavailable: {exc}")

        if importlib.util.find_spec("opendataloader_pdf") is None:
            warnings.append("opendataloader_pdf package is not installed")

        self._cached = PreflightResult(available=not warnings, warnings=warnings)
        return self._cached


class PdfAttachmentPreprocessor:
    def __init__(
        self,
        config: PdfPreprocessConfig | None = None,
        downloader: PdfDownloader | None = None,
        converter: PdfConverter | None = None,
        preflight: PreflightChecker | None = None,
    ) -> None:
        self.config = config or PdfPreprocessConfig.from_env()
        self.downloader = downloader or HttpxPdfDownloader()
        self.converter = converter or OpenDataLoaderConverter()
        self.preflight = preflight or OpenDataLoaderPreflight()

    def check_preflight(self) -> PreflightResult:
        if not self.config.enabled:
            return PreflightResult(available=False, warnings=["PDF preprocessing is disabled"])
        return self.preflight.check()

    async def preprocess(self, urls: list[str]) -> PdfPreprocessResult:
        if not self.config.enabled:
            return PdfPreprocessResult(fallback_urls=list(urls))

        warnings: list[str] = []
        preflight = self.preflight.check()
        if not preflight.available:
            warnings.extend(preflight.warnings)
            return PdfPreprocessResult(fallback_urls=list(urls), warnings=warnings)

        fallback_urls: list[str] = []
        downloaded: list[DownloadedPdf] = []

        with tempfile.TemporaryDirectory(prefix="campus-navi-pdf-") as temp_root:
            temp_dir = Path(temp_root)
            input_dir = temp_dir / "input"
            output_dir = temp_dir / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            for url in urls:
                if len(downloaded) >= self.config.max_files:
                    fallback_urls.append(url)
                    warnings.append(f"PDF preprocessing skipped max_files limit: {url}")
                    continue

                try:
                    downloaded_pdf = await self.downloader.download_pdf(
                        url,
                        input_dir,
                        max_bytes=self.config.max_bytes,
                        timeout_seconds=self.config.download_timeout_seconds,
                    )
                    downloaded.append(downloaded_pdf)
                except UnsupportedAttachmentError as exc:
                    fallback_urls.append(url)
                    warnings.append(f"PDF preprocessing skipped {url}: {exc}")
                except PdfDownloadError as exc:
                    fallback_urls.append(url)
                    warnings.append(f"PDF preprocessing download failed {url}: {exc}")

            if not downloaded:
                return PdfPreprocessResult(fallback_urls=fallback_urls, warnings=warnings)

            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        self.converter.convert,
                        [item.path for item in downloaded],
                        output_dir,
                        self.config,
                    ),
                    timeout=self.config.convert_timeout_seconds,
                )
            except PdfConverterError as exc:
                fallback_urls.extend(item.url for item in downloaded)
                warnings.append(str(exc))
                return PdfPreprocessResult(fallback_urls=fallback_urls, warnings=warnings)
            except asyncio.TimeoutError:
                fallback_urls.extend(item.url for item in downloaded)
                warnings.append("OpenDataLoader conversion timed out")
                return PdfPreprocessResult(fallback_urls=fallback_urls, warnings=warnings)

            extracted = _read_extracted_markdown(downloaded, output_dir, self.config)
            extracted_urls = {item.url for item in extracted}
            for item in downloaded:
                if item.url not in extracted_urls:
                    fallback_urls.append(item.url)
                    warnings.append(f"PDF preprocessing produced no markdown: {item.url}")

            return PdfPreprocessResult(
                extracted=extracted,
                fallback_urls=fallback_urls,
                warnings=warnings,
            )


def _read_extracted_markdown(
    downloaded: list[DownloadedPdf],
    output_dir: Path,
    config: PdfPreprocessConfig,
) -> list[ExtractedPdfAttachment]:
    markdown_by_stem = {
        path.stem: path
        for path in output_dir.rglob("*")
        if path.suffix.lower() in {".md", ".markdown"}
    }
    remaining_chars = config.extracted_text_max_chars
    extracted: list[ExtractedPdfAttachment] = []
    for item in downloaded:
        if remaining_chars <= 0:
            break

        markdown_path = markdown_by_stem.get(item.path.stem)
        if markdown_path is None:
            continue

        markdown = markdown_path.read_text(encoding="utf-8", errors="replace").strip()
        if not markdown:
            continue

        if len(markdown) > remaining_chars:
            markdown = f"{markdown[:remaining_chars]}\n\n[PDF markdown truncated]"
        remaining_chars -= len(markdown)

        extracted.append(
            ExtractedPdfAttachment(
                url=item.url,
                filename=item.filename,
                markdown=markdown,
            )
        )

    return extracted


def safe_filename_from_url(url: str, *, default: str = "attachment") -> str:
    raw_name = unquote(urlparse(url).path.rsplit("/", 1)[-1])
    name = raw_name or default
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    name = re.sub(r"_+(\.[A-Za-z0-9]+)$", r"\1", name)
    return name or default


def _has_clear_non_pdf_extension(url: str) -> bool:
    path = unquote(urlparse(url).path)
    filename = path.rsplit("/", 1)[-1]
    if "." not in filename:
        return False
    return f".{filename.rsplit('.', 1)[-1].lower()}" != ".pdf"


def _normalize_hybrid(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if normalized.lower() in {"", "off", "none", "false", "0"}:
        return None
    return normalized


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise PdfDownloadError(f"could not create unique filename for {path.name}")


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.lower() in {"1", "true", "yes", "on"}


def _env_hybrid(name: str, default: str | None) -> str | None:
    if name not in os.environ:
        return default
    return _normalize_hybrid(os.environ[name])


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return float(raw_value)


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return int(raw_value)
