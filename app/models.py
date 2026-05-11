from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ApplyMethodType = Literal["EMAIL", "OFFLINE", "PORTAL", "LINK", "OTHER"]


class OfficialProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    post_id: int | None = None
    structured_text: str
    image_urls: list[str] = Field(default_factory=list)
    attachment_urls: list[str] = Field(default_factory=list)


class OfficialProcessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    target_grade_min: int | None
    target_grade_max: int | None
    tag_code: str
    keywords: list[str] | None
    contact_phone: str | None
    contact_email: str | None
    start_date: str | None
    start_time: str | None
    end_date: str | None
    end_time: str | None
    required_documents: str | None
    apply_method_type: ApplyMethodType | None
    apply_method_detail: str | None
    eligibility: str | None
    is_applicable: bool


class BatchProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[OfficialProcessRequest]


class BatchProcessItemResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    post_id: int | None
    success: bool
    reason: str | None
    result: OfficialProcessResponse | None


class BatchProcessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[BatchProcessItemResult]
