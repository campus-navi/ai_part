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

```bash
.venv/bin/uvicorn app.main:app --reload
```

## API

- `POST /ai/official/process`: 공지 1건 분석
- `POST /ai/official/process/batch`: 공지 여러 건을 항목별 성공/실패로 분석

분석기는 현재 규칙 기반으로 동작합니다. 추후 Spring의 `GET /internal/tags`가
구현되면 태그 목록을 캐싱해 `tag_code` 분류에 활용하도록 확장할 수 있습니다.

## 테스트

```bash
.venv/bin/python -m pytest
```