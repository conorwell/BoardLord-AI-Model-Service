import pytest
from app import utils as tb_util

VALID_REQUEST = {
    "holds": [
        {"position_id": 380, "role": 5},  # start
        {"position_id": 452, "role": 6},  # hand
        {"position_id": 379, "role": 8},  # foot
        {"position_id": 758, "role": 7},  # finish
    ],
    "angle": 40,
    "is_nomatch": False,
}


class TestPredictEndpoint:
    def test_returns_200(self, client):
        response = client.post("/predict", json=VALID_REQUEST)
        assert response.status_code == 200

    def test_response_shape(self, client):
        data = client.post("/predict", json=VALID_REQUEST).json()
        assert "grade" in data
        assert "grade_class" in data
        assert "confidence" in data
        assert "probabilities" in data

    def test_grade_class_in_valid_range(self, client):
        data = client.post("/predict", json=VALID_REQUEST).json()
        assert 0 <= data["grade_class"] <= tb_util.NUM_CLASSES - 1

    def test_grade_label_matches_class(self, client):
        data = client.post("/predict", json=VALID_REQUEST).json()
        assert data["grade"] == tb_util.vgrade_to_label(data["grade_class"])

    def test_confidence_is_probability(self, client):
        data = client.post("/predict", json=VALID_REQUEST).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self, client):
        data = client.post("/predict", json=VALID_REQUEST).json()
        total = sum(data["probabilities"].values())
        assert abs(total - 1.0) < 1e-3

    def test_probabilities_has_all_grade_labels(self, client):
        data = client.post("/predict", json=VALID_REQUEST).json()
        expected = {tb_util.vgrade_to_label(i) for i in range(tb_util.NUM_CLASSES)}
        assert set(data["probabilities"].keys()) == expected

    def test_is_nomatch_accepted(self, client):
        req = {**VALID_REQUEST, "is_nomatch": True}
        response = client.post("/predict", json=req)
        assert response.status_code == 200

    def test_is_nomatch_defaults_to_false(self, client):
        req = {k: v for k, v in VALID_REQUEST.items() if k != "is_nomatch"}
        response = client.post("/predict", json=req)
        assert response.status_code == 200


class TestHealthEndpoint:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True
