# ai_part

FastAPI 기반 공지 분석 API입니다. `Spring, FastAPI API 문서.md`의 계약에 맞춰
단건 공지 분석과 배치 처리 엔드포인트를 제공합니다.

## Python 환경

현재 개발 환경은 안정 버전인 Python 3.12 가상환경을 사용합니다.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

## 실행

OpenAI API 키를 환경변수로 설정한 뒤 실행합니다.

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4.1-mini"
```

```bash
.venv/bin/uvicorn app.main:app --reload
```

## API

- `POST /ai/official/process`: 공지 1건 분석
- `POST /ai/official/process/batch`: 공지 여러 건을 항목별 성공/실패로 분석

분석기는 OpenAI Responses API로 `structured_text`, `image_urls`,
`attachment_urls`를 함께 전달해 멀티모달로 분석합니다. 응답은 Pydantic 스키마로
검증하며, `tag_code`는 Spring 백엔드의 seed 태그 코드(`COURSE`, `ACADEMIC`,
`ACTIVITY`, `SCHOLARSHIP`, `FACILITY`, `STUDENT_SUPPORT`) 중 하나를 반환합니다.
신청 관련 공지는 `apply_method_type`(`EMAIL`, `OFFLINE`, `PORTAL`, `LINK`,
`OTHER`)과 `apply_method_detail`, `is_applicable`을 함께 반환합니다.

환경변수:

- `OPENAI_API_KEY`: 필수
- `OPENAI_MODEL`: 선택, 기본값 `gpt-4.1-mini`
- `AI_MAX_IMAGES`: 선택, 기본값 `8`
- `AI_MAX_ATTACHMENTS`: 선택, 기본값 `8`
- `AI_IMAGE_DETAIL`: 선택, 기본값 `auto`

## 테스트

```bash
.venv/bin/python -m pytest
```
