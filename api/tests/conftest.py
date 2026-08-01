import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """No test may read the developer's real credentials.

    Uses monkeypatch throughout so every change is undone after each test —
    setting os.environ directly would leak across tests.
    """
    for key in (
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "ENABLE_ORPHEUS",
        "CACHE_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")


@pytest.fixture
def api_app():
    """A fresh application per test.

    The per-IP window lives on the application and every TestClient reports the
    same address, so a shared instance would let one module's requests exhaust
    another module's allowance — an ordering bug that looks like a flaky test.
    The limit is raised out of the way here; tests that exercise the window
    build their own app with a real one.
    """
    from newsninja.api.app import create_app
    from newsninja.config import Settings

    return create_app(Settings(rate_limit_per_minute=10_000))
