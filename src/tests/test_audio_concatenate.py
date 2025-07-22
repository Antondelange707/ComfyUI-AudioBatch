"""
Regression tests for the AudioConcatenate node in ComfyUI-AudioBatch.
"""

import pytest
import torch

import bootstrap  # noqa: F401
from nodes.nodes_audio import AudioConcatenate


# Helper function
def create_dummy_audio(batch_size, channels, samples, sr, device='cpu'):
    waveform = torch.linspace(0.1, 0.9, samples).repeat(batch_size, channels, 1)
    return {"waveform": waveform, "sample_rate": sr}


@pytest.fixture
def concat_node():
    return AudioConcatenate()


def test_concatenate_simple(concat_node):
    """Tests concatenating two perfectly matched audio clips."""
    sr = 44100
    samples1, samples2 = 1000, 2000
    audio1 = create_dummy_audio(1, 2, samples1, sr)
    audio2 = create_dummy_audio(1, 2, samples2, sr)

    (result_audio,) = concat_node.concatenate_audio(audio1, audio2)

    # Assertions
    assert result_audio['sample_rate'] == sr
    assert result_audio['waveform'].shape[0] == 1  # Batch size
    assert result_audio['waveform'].shape[1] == 2  # Channels
    assert result_audio['waveform'].shape[2] == samples1 + samples2  # Length

    # Check content
    assert torch.allclose(result_audio['waveform'][..., :samples1], audio1['waveform'])
    assert torch.allclose(result_audio['waveform'][..., samples1:], audio2['waveform'])


def test_concatenate_with_alignment(concat_node):
    """Tests that alignment (SR, channels) happens correctly before concatenation."""
    # Audio1: Stereo, 44.1k SR, 1s long
    sr1, len1 = 44100, 44100
    audio1 = create_dummy_audio(1, 2, len1, sr1)

    # Audio2: Mono, 22.05k SR, 1s long
    sr2, len2 = 22050, 22050
    audio2 = create_dummy_audio(1, 1, len2, sr2)

    (result_audio,) = concat_node.concatenate_audio(audio1, audio2)

    # Expected result:
    # SR = 44100
    # Channels = 2
    # Length = len1 + (len2 * sr1/sr2) = 44100 + (22050 * 2) = 88200
    expected_len = len1 + int(len2 * (sr1 / sr2))

    assert result_audio['sample_rate'] == sr1
    assert result_audio['waveform'].shape == (1, 2, expected_len)


def test_concatenate_batch_mismatch(concat_node):
    """Tests that batch mismatch is handled by repeating the last item."""
    sr = 44100
    samples = 1000
    # audio1 has 2 items, audio2 has 1 item
    audio1 = create_dummy_audio(2, 1, samples, sr)
    audio2 = create_dummy_audio(1, 1, samples, sr)

    (result_audio,) = concat_node.concatenate_audio(audio1, audio2)

    # Output batch should be max(2, 1) = 2
    assert result_audio['waveform'].shape[0] == 2
    assert result_audio['waveform'].shape[2] == samples * 2  # Concatenated length

    # The second item's second half should be a repeat of audio2's only item
    # First, let's get the second half of the second batch item
    second_item_second_half = result_audio['waveform'][1, :, samples:]

    # This should be equal to audio2's (only) item
    audio2_item = audio2['waveform'][0, :, :]

    assert torch.allclose(second_item_second_half, audio2_item)
