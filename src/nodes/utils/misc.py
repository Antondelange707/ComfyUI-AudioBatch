# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: GPL-3.0
# Project: ComfyUI-AudioBatch
import re


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


# Reference frequency for A4, the standard tuning pitch
A4_FREQ = 440.0
NOTES = {
    'c': -9, 'c#': -8, 'db': -8,
    'd': -7, 'd#': -6, 'eb': -6,
    'e': -5,
    'f': -4, 'f#': -3, 'gb': -3,
    'g': -2, 'g#': -1, 'ab': -1,
    'a': 0,  'a#': 1,  'bb': 1,
    'b': 2,
}


def parse_note_to_frequency(note_str: str, octave: int) -> float:
    """
    Parses a musical note string (e.g., `C#`, `A flat`, `db`) and an octave
    to calculate its frequency in Hz.

    Args:
        note_str (str): The note name. Case-insensitive. Handles sharps (  #), flats (b),
                        and text (`sharp`, `flat`).
        octave (int): The octave number (e.g., 4 for middle C's octave).

    Returns:
        float: The frequency of the note in Hz.

    Raises:
        ValueError: If the note name is invalid.
    """
    if not isinstance(note_str, str):
        raise TypeError("Note name must be a string.")

    # Normalize the string: lowercase, remove "sharp" or "flat" text, remove spaces
    processed_str = note_str.lower().strip()
    processed_str = re.sub(r'\s*sharp\s*', '#', processed_str)
    processed_str = re.sub(r'\s*flat\s*', 'b', processed_str)
    processed_str = processed_str.replace(" ", "")

    if processed_str not in NOTES:
        raise ValueError(f"Invalid note name: '{note_str}'. Could not parse to a valid note.")

    # Get the number of semitones away from A
    semitone_offset_from_a = NOTES[processed_str]

    # Calculate the number of semitones away from A4 (A in the 4th octave)
    # The note "A" in octave 4 is our base (0 semitones from itself).
    # The note "A" in octave 5 is 12 semitones higher.
    # The note "C" in octave 4 is -9 semitones from A4.
    n = semitone_offset_from_a + (octave - 4) * 12

    # Apply the frequency formula: f = f_base * (2^(1/12))^n
    frequency = A4_FREQ * (2**(1/12))**n

    return frequency
