import time

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from app.schemas import PredictRequest, PredictResponse
from app import analytics, predictor

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(request: Request, req: PredictRequest, background_tasks: BackgroundTasks):
    if not predictor.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    client_id = request.headers.get("X-Client-ID", "unknown")
    session_id = request.headers.get("X-Session-ID", "unknown")
    holds = [{"position_id": h.position_id, "role": h.role} for h in req.holds]

    start = time.monotonic()
    try:
        result = predictor.predict(holds, req.angle, req.is_nomatch)
    except RuntimeError as e:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        background_tasks.add_task(
            analytics.log_prediction,
            client_id=client_id,
            session_id=session_id,
            angle=req.angle,
            holds=holds,
            is_nomatch=req.is_nomatch,
            predicted_grade="",
            predicted_grade_class=-1,
            confidence=0.0,
            response_time_ms=elapsed_ms,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = int((time.monotonic() - start) * 1000)
    background_tasks.add_task(
        analytics.log_prediction,
        client_id=client_id,
        session_id=session_id,
        angle=req.angle,
        holds=holds,
        is_nomatch=req.is_nomatch,
        predicted_grade=result["grade"],
        predicted_grade_class=result["grade_class"],
        confidence=result["confidence"],
        response_time_ms=elapsed_ms,
    )

    return PredictResponse(**result)


@router.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor.is_loaded()}
