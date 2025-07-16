# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: GPL-3.0
# Project: ComfyUI-AudioBatch

NODES_NAME = "AudioBatch"
NODES_DEBUG_VAR = NODES_NAME.upper() + "_NODES_DEBUG"


def parse_time_to_seconds(time_str: str) -> float:
    """
    Converts a flexible time string into total seconds.

    Handles multiple formats:
    - A raw number of seconds: '123.45'
    - Seconds: '45.5'
    - Minutes and Seconds: '10:30.5'
    - Hours, Minutes, and Seconds: '01:10:30.5'

    Args:
        time_str: The string representing time.

    Returns:
        The total number of seconds as a float.

    Raises:
        ValueError: If the string format is invalid.
    """
    if time_str is None:
        return 0.0
    if not isinstance(time_str, str):
        raise TypeError("Input must be a string.")
    if not time_str:
        return 0.0

    # First, try a direct float conversion for the simplest case.
    try:
        return float(time_str)
    except ValueError:
        # If it fails, it contains non-numeric characters, likely ':'.
        pass

    # Scalable parsing for HH:MM:SS formats
    try:
        parts = time_str.split(':')
        total_seconds = 0.0
        multiplier = 1  # Starts with seconds

        # Iterate through parts in reverse (from seconds to hours)
        for part in reversed(parts):
            if multiplier > 3600:
                raise ValueError()
            if not part:  # Handles empty parts like in "1::30"
                raise RuntimeError("Empty part in time string (`{time_str}`)")
            part_number = float(part)
            if (part_number > 59 and multiplier < 3600) or part_number < 0:
                raise RuntimeError("Minutes or seconds segment is out of range (0-59) (`{time_str}`)")
            total_seconds += part_number * multiplier
            multiplier *= 60  # Next multiplier is 60 times bigger

        return total_seconds
    except (ValueError, TypeError) as e:
        # Re-raise with a clear, user-friendly message.
        raise ValueError(f"Invalid time format: '{time_str}'. "
                         "Expected 'SECONDS', 'MM:SS.ss', or 'HH:MM:SS.ss'.") from e
    except RuntimeError as e:
        raise ValueError(str(e))
