from newsninja.cache import Cache


def test_get_returns_none_for_a_missing_key(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    assert cache.get("nope") is None


def test_set_then_get_round_trips(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    cache.set("k", '{"a": 1}')
    assert cache.get("k") == '{"a": 1}'


def test_survives_reopening(tmp_path):
    path = tmp_path / "c.sqlite"
    Cache(path).set("k", "v")
    assert Cache(path).get("k") == "v"


def test_make_key_is_stable_regardless_of_argument_order():
    a = Cache.make_key(topic="ai", model="m", prompt_version="1")
    b = Cache.make_key(prompt_version="1", model="m", topic="ai")
    assert a == b


def test_make_key_changes_when_prompt_version_changes():
    a = Cache.make_key(topic="ai", model="m", prompt_version="1")
    b = Cache.make_key(topic="ai", model="m", prompt_version="2")
    assert a != b


def test_clear_empties_the_cache(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    cache.set("k", "v")
    cache.clear()
    assert cache.get("k") is None


def test_creates_parent_directories(tmp_path):
    cache = Cache(tmp_path / "nested" / "deeper" / "c.sqlite")
    cache.set("k", "v")
    assert cache.get("k") == "v"
