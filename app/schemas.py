from pydantic import BaseModel


class Hold(BaseModel):
    position_id: int
    role: int  # 5=start, 6=hand, 7=finish, 8=foot


class PredictRequest(BaseModel):
    holds: list[Hold]
    angle: int        # board angle in degrees, multiple of 5 (0–65)
    is_nomatch: bool = False


class PredictResponse(BaseModel):
    grade: str           # e.g. "V6", "≤V2", "V11+"
    grade_class: int     # model class index 0–10
    confidence: float    # softmax probability of predicted class
    probabilities: dict[str, float]  # all class labels → probability
