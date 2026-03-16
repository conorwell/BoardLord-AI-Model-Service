import pytest
from unittest.mock import patch
from starlette.testclient import TestClient

import app.predictor as predictor_mod
from app.main import app
from app.board_cnn import BoardCNN

# Minimal position map: position_id → (row, col) within the 35×33 grid.
# Uses three holds from a real example problem (silly_string) so role
# assignments are realistic.
MOCK_POS_MAP = {
    380: (10, 8),   # start hold
    452: (15, 12),  # hand hold
    758: (2, 16),   # finish hold
    379: (10, 7),   # foot hold
}

# position_id → 1.0 (plastic) or 0.0 (wood)
MOCK_MATERIAL_MAP = {
    380: 0.0,
    452: 1.0,
    758: 0.0,
    379: 0.0,
}


def _mock_load():
    """Seeds predictor module state with a random model and test data maps."""
    m = BoardCNN(dropout=0.0)
    m.eval()
    predictor_mod._model = m
    predictor_mod._pos_map = MOCK_POS_MAP
    predictor_mod._material_map = MOCK_MATERIAL_MAP


@pytest.fixture
def client():
    with patch("app.predictor.load", _mock_load):
        with TestClient(app) as c:
            yield c
