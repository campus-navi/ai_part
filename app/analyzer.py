from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

from app.models import OfficialProcessRequest, OfficialProcessResponse


TAG_CODES = (
    "COURSE",
    "ACADEMIC",
    "ACTIVITY",
    "SCHOLARSHIP",
    "FACILITY",
    "STUDENT_SUPPORT",
)

ANALYSIS_INSTRUCTIONS = f"""
You are the Campus Navi official-notice analysis engine.
Analyze Korean university notices using every provided source:
structured text, images, and attached files.

Return only data that is grounded in the notice. Do not guess missing details.
Follow these rules:
- `summary`: one concise Korean sentence.
- `tag_code`: choose exactly one of {", ".join(TAG_CODES)}.
- `keywords`: key Korean phrases from the notice, or null if none are useful.
- `target_grade_min` and `target_grade_max`: undergraduate grade range only. Use null if not specified.
- `is_applicable`: true only when the notice has a user action such as application, registration, submission, recruitment, or participation signup.
- If `is_applicable` is false, all application fields must be null:
  `start_date`, `start_time`, `end_date`, `end_time`, `required_documents`,
  `apply_method_type`, `apply_method_detail`, and `eligibility`.
- Application dates are application/submission period dates, not event dates. Use YYYY-MM-DD and HH:mm:ss.
- `apply_method_type`: EMAIL, OFFLINE, PORTAL, LINK, OTHER, or null.
- `apply_method_detail`: preserve concise actionable details such as portal name, URL, email address, office, or special instructions.
- `required_documents` and `eligibility`: preserve concise Korean text from the notice. Use newlines when multiple conditions are listed.
""".strip()


class NoticeAnalyzer(Protocol):
    async def analyze(self, request: OfficialProcessRequest) -> OfficialProcessResponse:
        ...


class AnalyzerConfigurationError(RuntimeError):
    pass


class AnalyzerExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAIAnalyzerConfig:
    model: str = "gpt-4.1-mini"
    max_images: int = 8
    max_attachments: int = 8
    image_detail: str = "auto"

    @classmethod
    def from_env(cls) -> OpenAIAnalyzerConfig:
        return cls(
            model=os.getenv("OPENAI_MODEL", cls.model),
            max_images=_env_int("AI_MAX_IMAGES", cls.max_images),
            max_attachments=_env_int("AI_MAX_ATTACHMENTS", cls.max_attachments),
            image_detail=os.getenv("AI_IMAGE_DETAIL", cls.image_detail),
        )


class OpenAINoticeAnalyzer:
    def __init__(self, config: OpenAIAnalyzerConfig | None = None) -> None:
        self.config = config or OpenAIAnalyzerConfig.from_env()
        self._client: Any | None = None

    async def analyze(self, request: OfficialProcessRequest) -> OfficialProcessResponse:
        if not request.structured_text.strip():
            raise ValueError("structured_text must not be blank")

        try:
            response = await self._get_client().responses.parse(
                model=self.config.model,
                instructions=ANALYSIS_INSTRUCTIONS,
                input=self._build_input(request),
                text_format=OfficialProcessResponse,
            )
        except AnalyzerConfigurationError:
            raise
        except Exception as exc:
            raise AnalyzerExecutionError(f"AI analysis failed: {exc}") from exc

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise AnalyzerExecutionError("AI response did not include parsed output")

        return OfficialProcessResponse.model_validate(parsed)

    def _build_input(self, request: OfficialProcessRequest) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": _format_notice_text(request),
            }
        ]

        for image_url in request.image_urls[: self.config.max_images]:
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_url,
                    "detail": self.config.image_detail,
                }
            )

        for attachment_url in request.attachment_urls[: self.config.max_attachments]:
            content.append(
                {
                    "type": "input_file",
                    "file_url": attachment_url,
                    "filename": _filename_from_url(attachment_url),
                }
            )

        return [{"role": "user", "content": content}]

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not os.getenv("OPENAI_API_KEY"):
            raise AnalyzerConfigurationError("OPENAI_API_KEY is required")

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise AnalyzerConfigurationError("openai package is not installed") from exc

        self._client = AsyncOpenAI()
        return self._client


def _format_notice_text(request: OfficialProcessRequest) -> str:
    return "\n".join(
        (
            f"post_id: {request.post_id}",
            "structured_text:",
            request.structured_text.strip(),
        )
    )


def _filename_from_url(url: str) -> str:
    path = urlparse(url).path
    filename = unquote(path.rsplit("/", 1)[-1])
    return filename or "attachment"


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise AnalyzerConfigurationError(f"{name} must be an integer") from exc

    if value < 0:
        raise AnalyzerConfigurationError(f"{name} must be greater than or equal to 0")

    return value
