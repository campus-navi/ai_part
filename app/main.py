from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from app.analyzer import (
    AnalyzerConfigurationError,
    AnalyzerExecutionError,
    NoticeAnalyzer,
    OpenAINoticeAnalyzer,
)
from app.models import (
    BatchProcessItemResult,
    BatchProcessRequest,
    BatchProcessResponse,
    OfficialProcessRequest,
    OfficialProcessResponse,
)


app = FastAPI(title="Campus Navi AI API")
notice_analyzer = OpenAINoticeAnalyzer()


def get_notice_analyzer() -> NoticeAnalyzer:
    return notice_analyzer


@app.post("/ai/official/process", response_model=OfficialProcessResponse)
async def process_official_notice(
    request: OfficialProcessRequest,
    analyzer: Annotated[NoticeAnalyzer, Depends(get_notice_analyzer)],
) -> OfficialProcessResponse:
    try:
        return await analyzer.analyze(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AnalyzerConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AnalyzerExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/ai/official/process/batch", response_model=BatchProcessResponse)
async def process_official_notice_batch(
    request: BatchProcessRequest,
    analyzer: Annotated[NoticeAnalyzer, Depends(get_notice_analyzer)],
) -> BatchProcessResponse:
    results: list[BatchProcessItemResult] = []

    for item in request.items:
        try:
            results.append(
                BatchProcessItemResult(
                    post_id=item.post_id,
                    success=True,
                    reason=None,
                    result=await analyzer.analyze(item),
                )
            )
        except Exception as exc:
            results.append(
                BatchProcessItemResult(
                    post_id=item.post_id,
                    success=False,
                    reason=f"AI 분석 오류: {exc}",
                    result=None,
                )
            )

    return BatchProcessResponse(results=results)
