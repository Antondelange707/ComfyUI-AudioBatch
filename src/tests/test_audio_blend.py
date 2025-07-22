"""
Regression tests for the AudioBlend node in ComfyUI-AudioBatch.
"""

import torch
import pytest
import logging

import bootstrap  # noqa: F401
# Now we can import the classes and functions to be tested
# We need AudioBlend, but it uses AudioBatchAligner internally, which is in the same file.
# We also import the logger to see node output during tests.
from nodes.nodes_audio import AudioBlend, logger as node_logger

# Configure logging for tests to see output from the node
node_logger.setLevel(logging.DEBUG)


# --- Pytest Fixtures (Reusable Setup) ---

@pytest.fixture
def audio_blend_node():
    """Provides an instance of the AudioBlend node for tests."""
    return AudioBlend()


# --- Helper Functions ---

def create_dummy_audio(batch_size: int, channels: int, samples: int, sr: int,
                       device: str = 'cpu') -> dict:
    """Creates a standard ComfyUI AUDIO dictionary for testing."""
    # Using torch.ones for predictable values, multiplied by a small float
    # to avoid pure 1s which can mask errors in blending.
    waveform = torch.ones(batch_size, channels, samples, device=device, dtype=torch.float32) * 0.5
    return {"waveform": waveform, "sample_rate": sr}


# --- Test Cases for AudioBlend ---

def test_blend_simple_match(audio_blend_node):
    """
    Tests blending two identical audio inputs.
    This is the "happy path" where no resampling, channel conversion, or padding is needed.
    """
    sr = 44100
    audio1 = create_dummy_audio(batch_size=1, channels=2, samples=1000, sr=sr)
    audio2 = create_dummy_audio(batch_size=1, channels=2, samples=1000, sr=sr)
    gain1, gain2 = 0.5, 0.8

    (result_audio,) = audio_blend_node.blend_audio(audio1, gain1, gain2, audio2)

    # Assertions
    expected_waveform = (audio1['waveform'] * gain1) + (audio2['waveform'] * gain2)
    assert result_audio['sample_rate'] == sr
    assert result_audio['waveform'].shape == expected_waveform.shape
    assert torch.allclose(result_audio['waveform'], expected_waveform)


def test_blend_no_audio2(audio_blend_node):
    """
    Tests the case where the optional `audio2` input is not provided (is None).
    """
    sr = 44100
    audio1 = create_dummy_audio(batch_size=2, channels=1, samples=1000, sr=sr)
    gain1, gain2 = 0.7, 1.0  # gain2 should be ignored

    (result_audio,) = audio_blend_node.blend_audio(audio1, gain1, gain2, audio2=None)

    # Assertions
    expected_waveform = audio1['waveform'] * gain1
    assert result_audio['sample_rate'] == sr
    assert result_audio['waveform'].shape == expected_waveform.shape
    assert torch.allclose(result_audio['waveform'], expected_waveform)


def test_blend_mismatched_sample_rate(audio_blend_node):
    """
    Tests if audio2 is correctly resampled to match audio1's sample rate.
    """
    sr1, sr2 = 44100, 22050
    audio1 = create_dummy_audio(batch_size=1, channels=2, samples=2000, sr=sr1)
    audio2 = create_dummy_audio(batch_size=1, channels=2, samples=1000, sr=sr2)
    gain1, gain2 = 1.0, 1.0

    (result_audio,) = audio_blend_node.blend_audio(audio1, gain1, gain2, audio2)

    # Assertions
    # Output SR should match audio1
    assert result_audio['sample_rate'] == sr1
    # Output length should match audio1, as it's longer after resampling audio2
    assert result_audio['waveform'].shape[2] == audio1['waveform'].shape[2]
    # Output batch and channels should match
    assert result_audio['waveform'].shape[0] == 1
    assert result_audio['waveform'].shape[1] == 2


def test_blend_mono_and_stereo(audio_blend_node):
    """
    Tests if a mono input is correctly converted to stereo when blended with a stereo input.
    """
    sr = 44100
    audio1_mono = create_dummy_audio(batch_size=1, channels=1, samples=1000, sr=sr)
    audio2_stereo = create_dummy_audio(batch_size=1, channels=2, samples=1000, sr=sr)
    gain1, gain2 = 1.0, 1.0

    (result_audio,) = audio_blend_node.blend_audio(audio1_mono, gain1, gain2, audio2_stereo)

    # Assertions
    assert result_audio['sample_rate'] == sr
    # Output should be stereo (2 channels)
    assert result_audio['waveform'].shape[1] == 2
    assert result_audio['waveform'].shape[2] == 1000

    # Check if the mono part was correctly duplicated and blended.
    # The two output channels should differ only by the difference in audio2's stereo channels.
    output_diff = result_audio['waveform'][:, 0, :] - result_audio['waveform'][:, 1, :]
    input2_diff = audio2_stereo['waveform'][:, 0, :] - audio2_stereo['waveform'][:, 1, :]
    # Since gain2 is 1.0, the difference should be the same.
    assert torch.allclose(output_diff, input2_diff)


def test_blend_mismatched_length(audio_blend_node):
    """
    Tests if the shorter audio is correctly padded to match the longer one.
    """
    sr = 44100
    len_short, len_long = 500, 1000
    audio1_short = create_dummy_audio(batch_size=1, channels=2, samples=len_short, sr=sr)
    audio2_long = create_dummy_audio(batch_size=1, channels=2, samples=len_long, sr=sr)
    gain1, gain2 = 1.0, 1.0

    (result_audio,) = audio_blend_node.blend_audio(audio1_short, gain1, gain2, audio2_long)

    # Assertions
    assert result_audio['sample_rate'] == sr
    # Output length should match the longer audio
    assert result_audio['waveform'].shape[2] == len_long

    # Check the tail end of the blended audio. It should only contain audio2's contribution.
    tail_start_index = len_short
    output_tail = result_audio['waveform'][..., tail_start_index:]
    expected_tail = audio2_long['waveform'][..., tail_start_index:] * gain2
    assert torch.allclose(output_tail, expected_tail)


def test_blend_mismatched_batch_size(audio_blend_node):
    """
    Tests blending inputs with different batch sizes. The output batch size should be the max.
    """
    sr = 44100
    b1, b2 = 2, 3
    samples = 1000
    channels = 2
    audio1 = create_dummy_audio(batch_size=b1, channels=channels, samples=samples, sr=sr)
    audio2 = create_dummy_audio(batch_size=b2, channels=channels, samples=samples, sr=sr)
    gain1, gain2 = 1.0, 1.0

    (result_audio,) = audio_blend_node.blend_audio(audio1, gain1, gain2, audio2)
    output_waveform = result_audio['waveform']

    # Assertions
    # Output batch size should be the max of the two inputs
    assert output_waveform.shape[0] == max(b1, b2)
    assert output_waveform.shape[1] == channels
    assert output_waveform.shape[2] == samples

    # The first two items should be a blend of audio1 and audio2
    expected_blend_part = (audio1['waveform'] * gain1) + (audio2['waveform'][:b1] * gain2)
    assert torch.allclose(output_waveform[:b1], expected_blend_part)

    # The last item (index 2) should only contain the contribution from audio2
    expected_last_item = audio2['waveform'][2] * gain2
    assert torch.allclose(output_waveform[2], expected_last_item)
