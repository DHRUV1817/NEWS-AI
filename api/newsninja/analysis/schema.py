"""Convert Pydantic JSON Schema into the strict dialect Groq requires.

Groq rejects Pydantic's default output. Verified against the live API on
2026-07-31, strict mode requires:

  * ``additionalProperties: false`` on every object, including under ``$defs``
  * every property listed in ``required`` (Pydantic omits defaulted fields)

``$ref``, ``$defs``, ``enum``, ``minimum`` and ``maximum`` pass through unchanged.
"""

import copy
from typing import Any

from pydantic import BaseModel


def _tighten(node: Any) -> None:
    """Recursively apply the strict-dialect rules in place."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            node["additionalProperties"] = False
            properties = node.get("properties", {})
            node["required"] = list(properties)
        for value in node.values():
            _tighten(value)
    elif isinstance(node, list):
        for item in node:
            _tighten(item)


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a strict-mode JSON Schema for ``model``.

    The caller's model is never mutated; the returned dict is a deep copy.
    """
    schema = copy.deepcopy(model.model_json_schema())
    _tighten(schema)
    return schema
