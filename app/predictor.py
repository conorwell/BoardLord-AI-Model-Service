import os
import torch
import torch.nn.functional as F
from google.cloud import storage

from app import utils as tb_util
from app.board_cnn import BoardCNN

DEVICE = torch.device("cpu")

_model: BoardCNN = None
_pos_map: dict = {}
_material_map: dict = {}


def load():
    global _model, _pos_map, _material_map

    bucket_name = os.environ.get("GCS_BUCKET")
    if not bucket_name:
        raise RuntimeError("GCS_BUCKET environment variable is not set")

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        model_path = "/tmp/model.pt"
        print(f"Downloading gs://{bucket_name}/models/board_cnn_latest.pt -> {model_path}")
        bucket.blob("models/board_cnn_latest.pt").download_to_filename(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download model from GCS: {e}") from e

    try:
        m = BoardCNN(dropout=0.0)
        m.load_state_dict(torch.load(model_path, map_location=DEVICE))
        m.eval()
        _model = m
        print("Model loaded.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}") from e

    _pos_map = tb_util.load_position_map()
    _material_map = tb_util.load_material_map()
    print("Static data loaded.")


def is_loaded() -> bool:
    return _model is not None


def predict(holds: list[dict], angle: int, is_nomatch: bool) -> dict:
    try:
        grid = tb_util.encode_climb(holds, _pos_map, _material_map).unsqueeze(0)

        angle_val = max(0, min(65, angle))
        angle_idx = torch.tensor([angle_val // 5], dtype=torch.long)
        nomatch = torch.tensor([0.0 if is_nomatch else 1.0], dtype=torch.float32)

        with torch.no_grad():
            logits = _model(grid, angle_idx, nomatch)
            probs = F.softmax(logits[0], dim=0)

        grade_class = int(probs.argmax().item())
        return {
            "grade": tb_util.vgrade_to_label(grade_class),
            "grade_class": grade_class,
            "confidence": round(float(probs[grade_class].item()), 4),
            "probabilities": {
                tb_util.vgrade_to_label(i): round(float(probs[i].item()), 4)
                for i in range(tb_util.NUM_CLASSES)
            },
        }
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}") from e
