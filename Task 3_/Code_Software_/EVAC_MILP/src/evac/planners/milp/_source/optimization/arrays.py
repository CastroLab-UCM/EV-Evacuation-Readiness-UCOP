"""Array shapes for MILP incidence construction."""

from __future__ import annotations
import numpy as np


def ragged_to_array(nested_list, *, fill=0, dtype=None) -> np.ndarray:
    max_dimensions = []
    inferred_type = int

    def scan(data, level=0):
        nonlocal inferred_type
        if isinstance(data, (list, tuple, np.ndarray)):
            if level >= len(max_dimensions):
                max_dimensions.append(0)
            max_dimensions[level] = max(max_dimensions[level], len(data))
            for item in data:
                scan(item, level + 1)
            return
        if isinstance(data, str):
            inferred_type = str
        elif isinstance(data, (float, np.float64)) and inferred_type is not str:
            inferred_type = float
        elif isinstance(data, (int, np.int_)):
            return
        elif isinstance(data, (bool, np.bool_)):
            inferred_type = bool
        elif data is not None:
            raise TypeError(f"Unsupported data type found: {type(data)}")

    scan(nested_list)
    shape = tuple(max_dimensions)
    final_dtype = dtype if dtype is not None else inferred_type
    try:
        np.array([fill], dtype=final_dtype)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Fill value '{fill}' of type {type(fill).__name__} is incompatible with the determined dtype '{np.dtype(final_dtype).name}'."
        ) from exc
    result = np.full(shape, fill_value=fill, dtype=final_dtype)

    def populate(target, source):
        for index, item in enumerate(source):
            if index >= len(target):
                continue
            if isinstance(item, (list, tuple, np.ndarray)):
                populate(target[index], item)
            elif item is not None:
                target[index] = item

    populate(result, nested_list)
    return result
