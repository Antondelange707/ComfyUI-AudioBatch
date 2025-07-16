"""
Tests for miscellaneous utility functions found in utils/misc.py.
"""
import bootstrap  # noqa: F401
import pytest

# Import the functions to be tested
from nodes.utils.misc import parse_note_to_frequency, parse_time_to_seconds


# --- Test Cases for parse_time_to_seconds ---

# Use pytest.mark.parametrize to test many valid inputs efficiently
@pytest.mark.parametrize("time_str, expected_seconds", [
    # Basic cases
    ("10", 10.0),
    ("15.5", 15.5),
    ("0", 0.0),

    # MM:SS format
    ("1:30", 90.0),
    ("01:30", 90.0),
    ("00:45", 45.0),
    ("10:00", 600.0),

    # MM:SS.ss format
    ("1:30.5", 90.5),
    ("0:05.25", 5.25),

    # HH:MM:SS format
    ("1:00:00", 3600.0),
    ("01:00:00", 3600.0),
    ("0:10:20", 620.0),

    # HH:MM:SS.ss format
    ("1:01:01.5", 3661.5),
    ("00:00:00.123", 0.123),

    # Edge cases
    ("", 0.0),                  # Empty string should default to 0
    ("  30.5  ", 30.5),         # Should handle whitespace
    (None, 0.0),                # None should default to 0
])
def test_parse_time_valid_inputs(time_str, expected_seconds):
    """Tests various valid time string formats."""
    assert parse_time_to_seconds(time_str) == pytest.approx(expected_seconds)


# Use pytest.mark.parametrize for invalid inputs that should raise ValueError
@pytest.mark.parametrize("invalid_time_str, expected_error_msg_part", [
    # Invalid characters
    ("abc", "Invalid time format"),
    ("10s", "Invalid time format"),
    ("1:20s", "Invalid time format"),

    # Malformed structures
    ("1:2:3:4", "Invalid time format"),  # Too many colons
    ("1::30", "Empty part in time string"),   # Empty part
    (":", "Empty part in time string"),
    (":30", "Empty part in time string"),

    # Invalid numbers within segments
    ("1:70", "Minutes or seconds segment is out of range (0-59)"),  # Seconds > 59
    ("1:60", "Minutes or seconds segment is out of range (0-59)"),  # Seconds == 60
    ("70:30", "Minutes or seconds segment is out of range (0-59)"),  # Minutes > 59
    ("1:30:80", "Minutes or seconds segment is out of range (0-59)"),  # Seconds > 59 in HH:MM:SS
])
def test_parse_time_invalid_inputs_raise_error(invalid_time_str, expected_error_msg_part):
    """
    Tests that invalid time string formats raise a ValueError
    with an informative message.
    """
    with pytest.raises(ValueError) as excinfo:
        parse_time_to_seconds(invalid_time_str)

    # Check that the exception message contains the expected substring
    assert expected_error_msg_part in str(excinfo.value)


def test_parse_time_non_string_input_raises_error():
    """Tests that non-string, non-None inputs raise a TypeError."""
    with pytest.raises(TypeError):
        parse_time_to_seconds(123)  # Pass an integer

    with pytest.raises(TypeError):
        parse_time_to_seconds([1, 2, 3])  # Pass a list

    with pytest.raises(TypeError):
        parse_time_to_seconds({"time": "10"})  # Pass a dict


# --- Test Cases for parse_note_to_frequency ---

@pytest.mark.parametrize("note, octave, expected_freq", [
    # Reference notes
    ("A", 4, 440.0),      # A4 standard pitch
    ("C", 4, 261.63),     # Middle C
    ("C", 0, 16.35),      # Very low C

    # Different notations
    ("C#", 4, 277.18),    # C sharp
    ("c sharp", 4, 277.18),  # C sharp with text
    ("c  sharp", 4, 277.18),  # C sharp with extra space
    ("Db", 4, 277.18),    # D flat (same as C#)
    ("d flat", 4, 277.18),  # D flat with text
    ("d      b", 4, 277.18),  # D flat with extra space

    # Different octaves
    ("A", 5, 880.0),      # One octave higher
    ("A", 3, 220.0),      # One octave lower
])
def test_parse_note_valid_inputs(note, octave, expected_freq):
    """Tests that valid note strings are parsed to the correct frequencies."""
    assert parse_note_to_frequency(note, octave) == pytest.approx(expected_freq, rel=0.001)


@pytest.mark.parametrize("invalid_note_str", [
    "H",         # Not a valid note letter
    "C##",       # Double sharp not supported by this simple parser
    "Dbb",       # Double flat not supported
    "A Sharps",  # Invalid text
    "123",       # Not a note
    "",          # Empty string
])
def test_parse_note_invalid_notes_raise_error(invalid_note_str):
    """Tests that invalid note names raise ValueError."""
    with pytest.raises(ValueError, match="Invalid note name"):
        parse_note_to_frequency(invalid_note_str, 4)


def test_parse_note_invalid_type_raises_error():
    """Tests that non-string inputs raise TypeError."""
    with pytest.raises(TypeError):
        parse_note_to_frequency(123, 4)
    with pytest.raises(TypeError):
        parse_note_to_frequency(None, 4)
