from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import OfficialProcessRequest, OfficialProcessResponse


@dataclass(frozen=True)
class TagRule:
    code: str
    keywords: tuple[str, ...]


TAG_RULES: tuple[TagRule, ...] = (
    TagRule(
        "SCHOLARSHIP",
        ("교내장학금", "교외장학금", "성적우수장학금", "장학금", "지원금", "수혜", "scholarship"),
    ),
    TagRule("EMPLOYMENT", ("취업", "인턴", "채용공고", "채용", "employment")),
    TagRule("ACADEMIC", ("학사", "수강", "졸업", "등록", "휴학", "복학")),
    TagRule("EVENT", ("행사", "특강", "세미나", "설명회")),
)

KEYWORD_CANDIDATES: tuple[str, ...] = (
    "교내장학금",
    "교외장학금",
    "성적우수장학금",
    "성적증명서",
    "재학증명서",
    "장학금",
    "지원금",
    "채용공고",
    "인턴",
    "채용",
    "취업",
    "수강",
    "졸업",
    "등록",
    "휴학",
    "복학",
    "휴관",
)

APPLICATION_MARKERS: tuple[str, ...] = (
    "신청",
    "접수",
    "지원방법",
    "신청방법",
    "모집",
    "지원자격",
    "신청자격",
    "필요서류",
    "제출서류",
)

PHONE_PATTERN = re.compile(
    r"\b(?:\d{2,3}-\d{3,4}-\d{4}|\d{2,3}\)\s?\d{3,4}-\d{4})\b"
)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
DATE_PATTERN = re.compile(
    r"(?P<date>\d{4}[./-]\d{1,2}[./-]\d{1,2})(?:\s*(?P<time>\d{1,2}:\d{2}(?::\d{2})?))?"
)
GRADE_PATTERN = re.compile(r"(?<!\d)([1-6])\s*학년")


def analyze_notice(request: OfficialProcessRequest) -> OfficialProcessResponse:
    text = request.structured_text.strip()
    if not text:
        raise ValueError("structured_text must not be blank")

    is_application = _is_application_notice(text)
    application_fields = _extract_application_fields(text) if is_application else {}
    target_grade_min, target_grade_max = _extract_grade_range(text)

    return OfficialProcessResponse(
        summary=_summarize(text),
        target_grade_min=target_grade_min,
        target_grade_max=target_grade_max,
        tag_code=_classify_tag(text),
        keywords=_extract_keywords(text),
        contact_phone=_first_match(PHONE_PATTERN, text),
        contact_email=_first_match(EMAIL_PATTERN, text),
        start_date=application_fields.get("start_date"),
        start_time=application_fields.get("start_time"),
        end_date=application_fields.get("end_date"),
        end_time=application_fields.get("end_time"),
        required_documents=application_fields.get("required_documents"),
        apply_method=application_fields.get("apply_method"),
        eligibility=application_fields.get("eligibility"),
        recruitment_count=application_fields.get("recruitment_count"),
    )


def _summarize(text: str) -> str:
    compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
    first_sentence = re.match(r"(.+?(?:[.!?。]|입니다\.?))(?:\s|$)", compact)
    if first_sentence:
        return first_sentence.group(1).strip()
    return compact[:80]


def _extract_grade_range(text: str) -> tuple[int | None, int | None]:
    if "전학년" in text or "전체 학년" in text:
        return 1, 4

    grades = [int(match) for match in GRADE_PATTERN.findall(text)]
    if not grades:
        return None, None
    return min(grades), max(grades)


def _classify_tag(text: str) -> str:
    lower_text = text.lower()
    best_code = "GENERAL"
    best_score = 0

    for rule in TAG_RULES:
        score = sum(1 for keyword in rule.keywords if keyword.lower() in lower_text)
        if score > best_score:
            best_code = rule.code
            best_score = score

    return best_code


def _extract_keywords(text: str) -> list[str] | None:
    keywords: list[str] = []
    for candidate in KEYWORD_CANDIDATES:
        if candidate not in text:
            continue
        if any(candidate in existing and candidate != existing for existing in keywords):
            continue
        keywords.append(candidate)

    return keywords or None


def _is_application_notice(text: str) -> bool:
    return any(marker in text for marker in APPLICATION_MARKERS)


def _extract_application_fields(text: str) -> dict[str, str | None]:
    first_date, second_date = _extract_dates(text)
    return {
        "start_date": first_date[0] if first_date else None,
        "start_time": first_date[1] if first_date else None,
        "end_date": second_date[0] if second_date else None,
        "end_time": second_date[1] if second_date else None,
        "required_documents": _extract_labeled_value(text, ("필요서류", "제출서류", "구비서류")),
        "apply_method": _extract_labeled_value(text, ("지원방법", "신청방법", "접수방법")),
        "eligibility": _extract_labeled_value(text, ("지원자격", "신청자격")),
        "recruitment_count": _extract_labeled_value(
            text,
            ("모집인원", "선발인원", "모집 인원", "선발 인원"),
        ),
    }


def _extract_dates(
    text: str,
) -> tuple[tuple[str, str | None] | None, tuple[str, str | None] | None]:
    dates = [
        (_normalize_date(match.group("date")), _normalize_time(match.group("time")))
        for match in DATE_PATTERN.finditer(text)
    ]
    first = dates[0] if dates else None
    second = dates[1] if len(dates) > 1 else None
    return first, second


def _normalize_date(value: str) -> str:
    year, month, day = re.split(r"[./-]", value)
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _normalize_time(value: str | None) -> str | None:
    if value is None:
        return None
    hour, minute, *second = value.split(":")
    return f"{int(hour):02d}:{int(minute):02d}:{int(second[0]) if second else 0:02d}"


def _extract_labeled_value(text: str, labels: tuple[str, ...]) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        for label in labels:
            pattern = rf"^{re.escape(label)}\s*[:：]\s*(?P<value>.+)$"
            match = re.match(pattern, stripped)
            if match:
                return match.group("value").strip()
    return None


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(0).strip() if match else None
