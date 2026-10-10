import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nutrifit import create_app  # noqa: E402
from nutrifit.recommender import FoodModel  # noqa: E402


@pytest.fixture(scope="session")
def model():
    return FoodModel()


@pytest.fixture()
def client():
    app = create_app({"TESTING": True, "SECRET_KEY": "test"})
    return app.test_client()
