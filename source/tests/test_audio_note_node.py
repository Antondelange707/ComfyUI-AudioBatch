"""
Regression tests for the AudioMusicalNote node in ComfyUI-AudioBatch.
"""

import bootstrap  # noqa: F401
import pytest
from nodes.nodes_audio import AudioMusicalNote


@pytest.fixture
def note_node():
    return AudioMusicalNote()


def test_note_node_valid_input(note_node):
    """Tests the node's integration with the parser for a valid note."""
    note = "C"
    octave = 4

    (frequency,) = note_node.get_frequency(note, octave)

    assert frequency == pytest.approx(261.63, rel=0.001)


def test_note_node_invalid_input(note_node):
    """
    Tests the node's error handling for an invalid note.
    It should not crash and should return a default frequency.
    """
    note = "Z"  # Invalid note
    octave = 4

    # The node should catch the ValueError from the parser and return a default
    (frequency,) = note_node.get_frequency(note, octave)

    # Assert that it returned the default fallback frequency
    assert frequency == 440.0
