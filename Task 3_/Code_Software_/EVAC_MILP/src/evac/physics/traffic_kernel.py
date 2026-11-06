"""Pure numerical traffic kernels shared by Python and compiled execution."""

from __future__ import annotations
import math
from typing import Any
from numba.extending import register_jitable


@register_jitable
def _bisect_right(values: Any, target: float) -> int:
    low = 0
    high = len(values)
    while low < high:
        middle = (low + high) // 2
        if target < values[middle]:
            high = middle
        else:
            low = middle + 1
    return low


@register_jitable
def traffic_exit_time_kernel(
    times: Any,
    multipliers: Any,
    entry_time_seconds: float,
    free_flow_duration_seconds: float,
) -> float:
    """Evaluate the accepted FIFO traffic integral on validated numeric inputs."""
    required_progress = free_flow_duration_seconds
    if required_progress == 0.0:
        return entry_time_seconds
    if len(times) == 1:
        return entry_time_seconds + required_progress * multipliers[0]
    current = entry_time_seconds
    remaining = required_progress
    while True:
        segment_index = _bisect_right(times, current) - 1
        if segment_index >= len(times) - 1:
            return current + remaining * multipliers[-1]
        left_time = times[segment_index]
        right_time = times[segment_index + 1]
        left_multiplier = multipliers[segment_index]
        right_multiplier = multipliers[segment_index + 1]
        duration = right_time - left_time
        slope = (right_multiplier - left_multiplier) / duration
        fraction = (current - left_time) / duration
        current_multiplier = left_multiplier + fraction * (
            right_multiplier - left_multiplier
        )
        available_duration = right_time - current
        if slope == 0.0:
            available_progress = available_duration / current_multiplier
        else:
            end_multiplier = current_multiplier + slope * available_duration
            available_progress = math.log(end_multiplier / current_multiplier) / slope
        if remaining <= available_progress:
            if slope == 0.0:
                elapsed = remaining * current_multiplier
            else:
                elapsed = current_multiplier * math.expm1(slope * remaining) / slope
            return current + elapsed
        remaining -= available_progress
        current = right_time
