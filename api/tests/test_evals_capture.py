from evals.capture import capture
from newsninja.errors import SourceError
from newsninja.models import Article


class FakeSource:
    name = "google_news"

    def __init__(self, error=None):
        self._error = error
        self.calls: list[str] = []

    def available(self) -> bool:
        return True

    def fetch(self, topic, limit=8):
        self.calls.append(topic)
        if self._error:
            raise self._error
        return [
            Article(title=f"{topic} {i}", url=f"https://e.com/{topic}/{i}",
                    source="google_news", body=f"body {i}")
            for i in range(limit)
        ]


def test_capture_returns_one_record_per_topic():
    records = capture(["ai", "climate"], source=FakeSource(), limit=3)
    assert [r.topic for r in records] == ["ai", "climate"]
    assert all(len(r.articles) == 3 for r in records)


def test_capture_skips_a_topic_whose_fetch_fails():
    source = FakeSource(error=SourceError("google_news", "boom"))
    records = capture(["ai"], source=source, limit=3)
    assert records == []


def test_capture_records_are_independent_per_topic():
    records = capture(["ai", "climate"], source=FakeSource(), limit=2)
    assert records[0].articles[0].title.startswith("ai")
    assert records[1].articles[0].title.startswith("climate")
