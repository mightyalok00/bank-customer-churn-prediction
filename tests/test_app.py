from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def test_app_loads_without_exception():
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    assert not app.exception
    assert app.title[0].value == "Bank customer churn predictor"
    assert len(app.metric) >= 4


def test_default_customer_can_be_scored():
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    app.button(key="predict_submit").click().run()
    assert not app.exception
    assert any(metric.label == "Churn probability" for metric in app.metric)
    assert app.success or app.warning or app.error
