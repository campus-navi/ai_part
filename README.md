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
- `POST /ai/academic-plan/review`: 학업계획서 섹션별 첨삭, diff, 수정 이유 생성

학업계획서 첨삭 기준은 `skills/academic_plan_review/SKILL.md`에서 읽습니다.
섹션별 모범 사례가 쌓이면 MCP RAG 서버나 `retrieve_reference_examples` 쪽만 확장하면 됩니다.
AI Agent는 수정문과 수정 이유만 생성하고, `original_content`, `diff`, `changes`,
`inline_diff`는 서버가 계산해 최종 응답 JSON을 조립합니다.
700자 이상 원문은 수정문을 최소 650자, 원문 대비 80% 이상으로 유지하도록 검증하며,
Agent는 `count_korean_characters` tool로 글자 수를 확인할 수 있습니다.
섹션 누락을 막기 위해 Agent 실행은 섹션별로 한 번씩 수행하고 서버가 최종 응답을 합칩니다.

분석기는 OpenAI Responses API로 `structured_text`, `image_urls`,
`attachment_urls`를 함께 전달해 멀티모달로 분석합니다. 응답은 Pydantic 스키마로
검증하며, `tag_code`는 Spring 백엔드의 seed 태그 코드(`COURSE`, `ACADEMIC`,
`ACTIVITY`, `SCHOLARSHIP`, `FACILITY`, `STUDENT_SUPPORT`) 중 하나를 반환합니다.
신청 관련 공지는 `apply_method_type`(`FILE`, `OFFLINE`, `PORTAL`, `LINK`,
`OTHER`)과 `apply_method_detail`, `is_applicable`을 함께 반환합니다.

PDF 첨부는 OpenDataLoader로 Markdown/JSON 변환을 먼저 시도합니다. 성공한 PDF는
추출 Markdown을 AI 입력 텍스트에 포함합니다. HWP/HWPX 첨부는 `rhwp-python`으로
텍스트 추출을 먼저 시도하고, 성공한 본문을 AI 입력 텍스트에 포함합니다. 실패한
PDF와 OpenAI file inputs가 지원하는 문서/스프레드시트/프레젠테이션/텍스트·코드
첨부는 `input_file` URL로 전달하고, 이미지 첨부는 `input_image`로 전달합니다.
그 외 ZIP 등 지원 목록 밖의 첨부는 OpenAI 파일 입력에서 제외하고 파일명만
텍스트 힌트로 제공합니다.

환경변수:

- `OPENAI_API_KEY`: 필수
- `OPENAI_MODEL`: 선택, 기본값 `gpt-4.1-mini`
- `ACADEMIC_PLAN_MODEL`: 선택, 학업계획서 첨삭 Agent 모델, 기본값 `gpt-4.1-mini`
- `ACADEMIC_PLAN_MAX_TOKENS`: 선택, 학업계획서 첨삭 Agent 최대 출력 토큰, 기본값 `3000`
- `ACADEMIC_PLAN_MCP_COMMAND`: 선택, RAG MCP stdio 서버 실행 명령, 기본값 현재 Python
- `ACADEMIC_PLAN_MCP_ARGS`: 선택, MCP stdio 서버 인자, 기본값 `-m app.academic_plan_mcp`
- `ACADEMIC_PLAN_MCP_NAME`: 선택, MCP 서버 이름, 기본값 `academic-plan-rag`
- `AI_MAX_IMAGES`: 선택, 기본값 `8`
- `AI_MAX_ATTACHMENTS`: 선택, 기본값 `8`
- `AI_IMAGE_DETAIL`: 선택, 기본값 `auto`
- `PDF_PREPROCESS_ENABLED`: 선택, 기본값 `true`
- `PDF_DOWNLOAD_TIMEOUT_SECONDS`: 선택, 기본값 `10`
- `PDF_MAX_BYTES`: 선택, 기본값 `10485760`
- `PDF_MAX_FILES`: 선택, 기본값 `8`
- `PDF_EXTRACTED_TEXT_MAX_CHARS`: 선택, 기본값 `60000`
- `PDF_CONVERT_TIMEOUT_SECONDS`: 선택, 기본값 `180`
- `OPENDATALOADER_HYBRID`: 선택, 기본값 `docling-fast` (``, `off`, `none`이면 hybrid 비활성)
- `OPENDATALOADER_HYBRID_MODE`: 선택, 기본값 `full`
- `OPENDATALOADER_HYBRID_TIMEOUT_MS`: 선택, 기본값 `120000`
- `OPENDATALOADER_QUIET`: 선택, 기본값 `false`
- `OPENDATALOADER_HYBRID_FALLBACK`: 선택, 기본값 `true`
- `HWP_PREPROCESS_ENABLED`: 선택, 기본값 `true`
- `HWP_DOWNLOAD_TIMEOUT_SECONDS`: 선택, 기본값 `10`
- `HWP_MAX_BYTES`: 선택, 기본값 `10485760`
- `HWP_MAX_FILES`: 선택, 기본값 `8`
- `HWP_EXTRACTED_TEXT_MAX_CHARS`: 선택, 기본값 `60000`
- `HWP_EXTRACT_TIMEOUT_SECONDS`: 선택, 기본값 `60`

## 테스트

```bash
.venv/bin/python -m pytest
```

OpenAI API를 실제 호출하는 학업계획서 첨삭 smoke test:

```bash
RUN_OPENAI_API_TESTS=1 .venv/bin/python -m pytest -s tests/test_academic_plan_openai_api.py
```

이 테스트는 `application_motive`, `interest_field`, `study_plan`, `etc_info` 네 섹션을
모두 포함하며 각 섹션은 700자 이상 1,000자 이하입니다. 필요하면
`OPENAI_API_TEST_MAX_TOKENS`로 출력 토큰 한도를 조절합니다.
