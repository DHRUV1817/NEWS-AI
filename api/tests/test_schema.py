from newsninja.analysis.schema import strict_schema
from newsninja.models import ArticleAnalysis, Briefing


def _every_object(node):
    """Yield every JSON-Schema object node, including those under $defs."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for value in node.values():
            yield from _every_object(value)
    elif isinstance(node, list):
        for item in node:
            yield from _every_object(item)


def test_every_object_forbids_additional_properties():
    schema = strict_schema(ArticleAnalysis)
    objects = list(_every_object(schema))
    assert objects, "expected at least the root object"
    for obj in objects:
        assert obj["additionalProperties"] is False


def test_every_property_is_required():
    schema = strict_schema(ArticleAnalysis)
    for obj in _every_object(schema):
        assert set(obj["required"]) == set(obj["properties"])


def test_fields_with_defaults_become_required():
    """Pydantic omits defaulted fields from `required`; strict mode needs them."""
    schema = strict_schema(Briefing)
    assert "language" in schema["required"]


def test_defs_are_preserved():
    schema = strict_schema(ArticleAnalysis)
    assert "Entity" in schema["$defs"]
    assert schema["$defs"]["Entity"]["additionalProperties"] is False


def test_the_original_schema_is_not_mutated():
    original = ArticleAnalysis.model_json_schema()
    before = original.get("additionalProperties", "absent")
    strict_schema(ArticleAnalysis)
    assert original.get("additionalProperties", "absent") == before
