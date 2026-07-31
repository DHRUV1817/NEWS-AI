import pytest

from evals.corpus import CorpusRecord, load_corpus, save_corpus
from newsninja.models import Article


def _record(topic: str = "ai") -> CorpusRecord:
    return CorpusRecord(
        topic=topic,
        articles=[
            Article(title="A", url="https://e.com/1", source="google_news", body="body one"),
            Article(title="B", url="https://e.com/2", source="google_news", body="body two"),
        ],
    )


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "corpus.jsonl"
    save_corpus([_record("ai"), _record("climate")], path)
    loaded = load_corpus(path)
    assert [r.topic for r in loaded] == ["ai", "climate"]
    assert loaded[0].articles[0].body == "body one"


def test_load_missing_file_raises_with_a_usable_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="capture"):
        load_corpus(tmp_path / "absent.jsonl")


def test_save_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "corpus.jsonl"
    save_corpus([_record("ai"), _record("climate")], path)
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2


def test_empty_corpus_round_trips(tmp_path):
    path = tmp_path / "corpus.jsonl"
    save_corpus([], path)
    assert load_corpus(path) == []
