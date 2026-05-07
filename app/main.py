from fastapi import FastAPI, HTTPException

from app.analyzer import analyze_notice
from app.models import (
    BatchProcessItemResult,
    BatchProcessRequest,
    BatchProcessResponse,
    OfficialProcessRequest,
    OfficialProcessResponse,
)


app = FastAPI(title="Campus Navi AI API")


@app.post("/ai/official/process", response_model=OfficialProcessResponse)
async def process_official_notice(request: OfficialProcessRequest) -> OfficialProcessResponse:
    try:
        return analyze_notice(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ai/official/process/batch", response_model=BatchProcessResponse)
async def process_official_notice_batch(request: BatchProcessRequest) -> BatchProcessResponse:
    results: list[BatchProcessItemResult] = []

    for item in request.items:
        try:
            results.append(
                BatchProcessItemResult(
                    post_id=item.post_id,
                    success=True,
                    reason=None,
                    result=analyze_notice(item),
                )
            )
        except Exception as exc:
            results.append(
                BatchProcessItemResult(
                    post_id=item.post_id,
                    success=False,
                    reason=f"텍스트 파싱 오류: {exc}",
                    result=None,
                )
            )

    return BatchProcessResponse(results=results)
