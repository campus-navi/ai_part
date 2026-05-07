from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_process_official_notice_extracts_document_contract_fields():
    response = client.post(
        "/ai/official/process",
        json={
            "post_id": 123,
            "structured_text": (
                "2026학년도 2학기 교내장학금 신청 안내입니다.\n"
                "대상: 2학년부터 4학년까지 재학생\n"
                "신청기간: 2026.04.01 09:00부터 2026-05-31까지\n"
                "필요서류: 성적증명서, 재학증명서\n"
                "지원방법: 포털 온라인 신청\n"
                "지원자격: 현재 3학기 이상 이수 중인 재학생\n"
                "모집인원: 100명\n"
                "문의: 02-3290-1234 scholarship@example.ac.kr"
            ),
            "image_urls": [],
            "attachment_urls": ["https://s3.amazonaws.com/example/notice.pdf"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "summary": "2026학년도 2학기 교내장학금 신청 안내입니다.",
        "target_grade_min": 2,
        "target_grade_max": 4,
        "tag_code": "SCHOLARSHIP",
        "keywords": ["교내장학금", "성적증명서", "재학증명서"],
        "contact_phone": "02-3290-1234",
        "contact_email": "scholarship@example.ac.kr",
        "start_date": "2026-04-01",
        "start_time": "09:00:00",
        "end_date": "2026-05-31",
        "end_time": None,
        "required_documents": "성적증명서, 재학증명서",
        "apply_method": "포털 온라인 신청",
        "eligibility": "현재 3학기 이상 이수 중인 재학생",
        "recruitment_count": "100명",
    }


def test_process_non_application_notice_returns_application_fields_as_null():
    response = client.post(
        "/ai/official/process",
        json={
            "post_id": 124,
            "structured_text": "도서관 임시 휴관 안내입니다. 전기 점검으로 5월 1일 휴관합니다.",
            "image_urls": [],
            "attachment_urls": [],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == "도서관 임시 휴관 안내입니다."
    assert body["tag_code"] == "GENERAL"
    assert body["start_date"] is None
    assert body["end_date"] is None
    assert body["required_documents"] is None
    assert body["apply_method"] is None
    assert body["eligibility"] is None
    assert body["recruitment_count"] is None


def test_batch_process_keeps_successes_when_one_item_fails():
    response = client.post(
        "/ai/official/process/batch",
        json={
            "items": [
                {
                    "post_id": 125,
                    "structured_text": "인턴 채용공고입니다. 신청기간: 2026-06-01부터 2026-06-10까지",
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
    assert results[125]["result"]["tag_code"] == "EMPLOYMENT"
    assert results[126]["success"] is False
    assert "structured_text" in results[126]["reason"]
    assert results[126]["result"] is None
