"""
Regression tests for the SelectAudioFromBatch node in ComfyUI-AudioBatch.
"""

import bootstrap  # noqa: F401
import torch
import pytest
from nodes.nodes_audio import SelectAudioFromBatch


# Helper function
def create_dummy_audio(batch_size, channels, samples, sr, device='cpu'):
    waveform = torch.randn(batch_size, channels, samples, device=device, dtype=torch.float32)
    return {"waveform": waveform, "sample_rate": sr}


@pytest.fixture
def select_node():
    return SelectAudioFromBatch()


def test_select_valid_index(select_node):
    """Tests selecting a valid index from the batch."""
    sr = 44100
    audio_batch = create_dummy_audio(5, 2, 1000, sr)

    (result_audio,) = select_node.select_audio(audio_batch, index=2, behavior_out_of_range="error",
                                               silence_duration_seconds=1.0)

    # Output should be a batch of 1
    assert result_audio['waveform'].shape[0] == 1
    # It should be the 3rd item from the original batch
    assert torch.allclose(result_audio['waveform'], audio_batch['waveform'][2:3, :, :])
    assert result_audio['sample_rate'] == sr


def test_select_out_of_range_error(select_node):
    """Tests if an error is raised when index is out of range and behavior is 'error'."""
    sr = 44100
    audio_batch = create_dummy_audio(3, 2, 1000, sr)

    with pytest.raises(ValueError, match="Index 5 is out of range for batch of size 3"):
        select_node.select_audio(audio_batch, index=5, behavior_out_of_range="error", silence_duration_seconds=1.0)


def test_select_out_of_range_silence_original_length(select_node):
    """Tests if silent audio of original length is returned for an out-of-range index."""
    sr = 44100
    samples = 1234
    channels = 2
    audio_batch = create_dummy_audio(3, channels, samples, sr)

    (result_audio,) = select_node.select_audio(audio_batch, index=3, behavior_out_of_range="silence_original_length",
                                               silence_duration_seconds=1.0)

    assert result_audio['waveform'].shape == (1, channels, samples)
    assert result_audio['sample_rate'] == sr
    # Check if it's all zeros (silence)
    assert torch.all(result_audio['waveform'] == 0)


def test_select_out_of_range_silence_fixed_length(select_node):
    """Tests if silent audio of a fixed length is returned for an out-of-range index."""
    sr = 48000
    audio_batch = create_dummy_audio(3, 1, 1000, sr)
    silence_duration = 2.5

    (result_audio,) = select_node.select_audio(audio_batch, index=10, behavior_out_of_range="silence_fixed_length",
                                               silence_duration_seconds=silence_duration)

    expected_samples = int(sr * silence_duration)

    assert result_audio['waveform'].shape == (1, 1, expected_samples)
    assert result_audio['sample_rate'] == sr
    assert torch.all(result_audio['waveform'] == 0)
