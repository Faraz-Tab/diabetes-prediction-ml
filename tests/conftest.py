import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

VALID = {"pregnancies": "2", "glucose": "148", "bloodpressure": "72", "skinthickness": "35",
         "insulin": "0", "bmi": "33.6", "dpf": "0.627", "age": "50"}


@pytest.fixture(scope="session")
def client():
    from app import create_app
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def valid():
    return dict(VALID)
