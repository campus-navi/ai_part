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

OpenAI API 키는 `.env`로 관리합니다. `.env.example`을 복사해 값을 채우고,
PDF 첨부 전처리를 위해 Java 11+와 OpenDataLoader hybrid backend를 준비한 뒤
실행합니다.

```bash
cp .env.example .env
# .env의 OPENAI_API_KEY 값을 채웁니다.
opendataloader-pdf-hybrid --port 5002 --force-ocr --ocr-lang "ko,en"
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
신청 관련 공지는 `apply_method_type`(`FILE`, `OFFLINE`, `PORTAL`, `LINK`,
`OTHER`)과 `apply_method_detail`, `is_applicable`을 함께 반환합니다.

PDF 첨부는 OpenDataLoader로 Markdown/JSON 변환을 먼저 시도합니다. 성공한 PDF는
추출 Markdown을 AI 입력 텍스트에 포함합니다. 실패한 PDF와 지원되는 문서 첨부는
OpenAI `input_file` URL로 전달하고, 이미지 첨부는 `input_image`로 전달합니다.
지원하지 않는 첨부는 warning 로그만 남기고 제외합니다.

환경변수:

- `OPENAI_API_KEY`: 필수
- `OPENAI_MODEL`: 선택, 기본값 `gpt-4.1-mini`
- `AI_MAX_IMAGES`: 선택, 기본값 `8`
- `AI_MAX_ATTACHMENTS`: 선택, 기본값 `8`
- `AI_IMAGE_DETAIL`: 선택, 기본값 `auto`
- `PDF_PREPROCESS_ENABLED`: 선택, 기본값 `true`
- `PDF_DOWNLOAD_TIMEOUT_SECONDS`: 선택, 기본값 `10`
- `PDF_MAX_BYTES`: 선택, 기본값 `10485760`
- `PDF_MAX_FILES`: 선택, 기본값 `8`
- `PDF_EXTRACTED_TEXT_MAX_CHARS`: 선택, 기본값 `60000`
- `PDF_CONVERT_TIMEOUT_SECONDS`: 선택, 기본값 `180`
- `OPENDATALOADER_HYBRID`: 선택, 기본값 `docling-fast`
- `OPENDATALOADER_HYBRID_MODE`: 선택, 기본값 `full`
- `OPENDATALOADER_HYBRID_TIMEOUT_MS`: 선택, 기본값 `120000`

## 테스트

```bash
.venv/bin/python -m pytest
```
