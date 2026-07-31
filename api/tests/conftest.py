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
