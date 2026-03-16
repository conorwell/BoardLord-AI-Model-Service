from fastapi import APIRouter
from app.schemas import PredictRequest, PredictResponse
from app import predictor

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    holds = [{"position_id": h.position_id, "role": h.role} for h in req.holds]
    result = predictor.predict(holds, req.angle, req.is_nomatch)
    return PredictResponse(**result)


@router.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor.is_loaded()}
