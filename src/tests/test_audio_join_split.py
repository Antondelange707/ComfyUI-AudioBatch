"""
Regression tests for the AudioJoin2Channels and AudioSplit2Channels nodes in ComfyUI-AudioBatch.
"""

import bootstrap  # noqa: F401
import torch
import pytest
from nodes.nodes_audio import AudioJoin2Channels, AudioSplit2Channels


# Helper function
def create_dummy_audio(batch_size, channels, samples, sr, device='cpu'):
    waveform = torch.randn(batch_size, channels, samples, device=device, dtype=torch.float32)
    return {"waveform": waveform, "sample_rate": sr}


@pytest.fixture
def join_node():
    return AudioJoin2Channels()


@pytest.fixture
def split_node():
    return AudioSplit2Channels()


# --- Tests for AudioJoin2Channels ---

def test_join_simple(join_node):
    """Tests joining two perfectly matched mono signals."""
    sr = 44100
    samples = 1000
    left_audio = create_dummy_audio(1, 1, samples, sr)
    right_audio = create_dummy_audio(1, 1, samples, sr)

    (stereo_audio,) = join_node.join_channels(left_audio, right_audio)

    assert stereo_audio['waveform'].shape == (1, 2, samples)
    # Check if L/R channels match the original mono inputs
    assert torch.allclose(stereo_audio['waveform'][:, 0:1, :], left_audio['waveform'])
    assert torch.allclose(stereo_audio['waveform'][:, 1:2, :], right_audio['waveform'])


def test_join_converts_inputs_to_mono(join_node):
    """Tests that stereo inputs are correctly converted to mono before joining."""
    sr = 44100
    samples = 1000
    # Left input is stereo, Right is mono
    left_stereo_in = create_dummy_audio(1, 2, samples, sr)
    right_mono_in = create_dummy_audio(1, 1, samples, sr)

    (result_audio,) = join_node.join_channels(left_stereo_in, right_mono_in)

    # Left channel of the output should be the average of the stereo input
    expected_left_mono = torch.mean(left_stereo_in['waveform'], dim=1, keepdim=True)

    assert result_audio['waveform'].shape == (1, 2, samples)
    assert torch.allclose(result_audio['waveform'][:, 0:1, :], expected_left_mono)
    assert torch.allclose(result_audio['waveform'][:, 1:2, :], right_mono_in['waveform'])


def test_join_aligns_and_batches(join_node):
    """Tests that mismatched SR, length, and batch sizes are handled."""
    # Left: B=2, 1ch, 44.1k SR, 1s long
    audio_left = create_dummy_audio(2, 1, 44100, 44100)
    # Right: B=3, 2ch, 22.05k SR, 0.5s long
    audio_right = create_dummy_audio(3, 2, 11025, 22050)

    (result_audio,) = join_node.join_channels(audio_left, audio_right)

    # Expected output shape:
    # Batch size = max(2, 3) = 3
    # Channels = 2 (stereo)
    # SR = 44100 (from left)
    # Samples = 44100 (from left, since right is 11025*2=22050 after resampling)
    assert result_audio['waveform'].shape == (3, 2, 44100)
    assert result_audio['sample_rate'] == 44100

    # Check the repeated last item for the left channel
    assert torch.allclose(result_audio['waveform'][1, 0, :], result_audio['waveform'][2, 0, :])


# --- Tests for AudioSplit2Channels ---

def test_split_simple(split_node):
    """Tests splitting a standard stereo signal."""
    sr = 44100
    samples = 1000
    stereo_audio = create_dummy_audio(1, 2, samples, sr)

    (left_out, right_out) = split_node.split_channels(stereo_audio)

    # Check left channel output
    assert left_out['sample_rate'] == sr
    assert left_out['waveform'].shape == (1, 1, samples)
    assert torch.allclose(left_out['waveform'], stereo_audio['waveform'][:, 0:1, :])

    # Check right channel output
    assert right_out['sample_rate'] == sr
    assert right_out['waveform'].shape == (1, 1, samples)
    assert torch.allclose(right_out['waveform'], stereo_audio['waveform'][:, 1:2, :])


def test_split_with_batch(split_node):
    """Tests that splitting preserves the batch dimension."""
    sr = 44100
    samples = 1000
    stereo_batch = create_dummy_audio(5, 2, samples, sr)

    (left_out, right_out) = split_node.split_channels(stereo_batch)

    assert left_out['waveform'].shape == (5, 1, samples)
    assert right_out['waveform'].shape == (5, 1, samples)


def test_split_raises_error_on_non_stereo(split_node):
    """Tests that an error is raised if the input is not stereo."""
    # Test with mono
    mono_audio = create_dummy_audio(1, 1, 1000, 44100)
    with pytest.raises(ValueError, match="Input audio must be stereo"):
        split_node.split_channels(mono_audio)

    # Test with 3 channels
    three_ch_audio = create_dummy_audio(1, 3, 1000, 44100)
    with pytest.raises(ValueError, match="Input audio must be stereo"):
        split_node.split_channels(three_ch_audio)
