import operator
from typing import Any


def override_reducer(current_value: Any, new_value: Any) -> Any:
    """Supports both append and full-replace semantics.
    
    If new_value is {"type": "override", "value": X}, replaces current_value with X.
    Otherwise appends new_value to current_value (list concatenation).
    """
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", [])
    if current_value is None:
        current_value = []
    if isinstance(new_value, list):
        return operator.add(current_value, new_value)
    return operator.add(current_value, [new_value])
