"""
Regression tests for the AudioNormalize and AudioApplyBatchedGain nodes in ComfyUI-AudioBatch.
"""

import torch
import pytest
import bootstrap  # noqa: F401
from nodes.nodes_audio import AudioNormalize, AudioApplyBatchedGain


# Helper function
def create_dummy_audio(batch_size, channels, samples, sr, peak_value=0.5, device='cpu'):
    # Create audio that is not normalized to 1.0 to make testing meaningful
    waveform = torch.randn(batch_size, channels, samples, device=device)
    # Normalize to a known peak value
    current_peak, _ = torch.max(torch.abs(waveform), dim=2, keepdim=True)
    current_peak, _ = torch.max(current_peak, dim=1, keepdim=True)
    waveform = waveform / (current_peak + 1e-9) * peak_value
    return {"waveform": waveform, "sample_rate": sr}


@pytest.fixture
def normalize_node():
    return AudioNormalize()


@pytest.fixture
def apply_gain_node():
    return AudioApplyBatchedGain()


# --- Tests for AudioNormalize ---

def test_normalize_simple(normalize_node):
    """Tests if a simple audio clip is normalized to the target peak level."""
    sr = 44100
    original_peak = 0.25
    target_peak = 1.0
    audio = create_dummy_audio(1, 1, sr, sr, peak_value=original_peak)

    (normalized_audio, original_peak_level) = normalize_node.normalize_audio(audio, target_peak)

    # Assertions
    new_peak = torch.max(torch.abs(normalized_audio['waveform']))
    assert new_peak.item() == pytest.approx(target_peak, abs=1e-6)
    assert original_peak_level.item() == pytest.approx(original_peak, abs=1e-6)


def test_normalize_with_batch(normalize_node):
    """Tests that each item in a batch is normalized independently."""
    sr = 44100
    # Create a batch with different peak levels
    audio1 = create_dummy_audio(1, 2, sr, sr, peak_value=0.5)
    audio2 = create_dummy_audio(1, 2, sr, sr, peak_value=0.1)

    batched_waveform = torch.cat((audio1['waveform'], audio2['waveform']), dim=0)
    batched_audio = {"waveform": batched_waveform, "sample_rate": sr}

    target_peak = 0.9
    (normalized_audio, original_peak_levels) = normalize_node.normalize_audio(batched_audio, target_peak)

    # Check item 1
    peak1 = torch.max(torch.abs(normalized_audio['waveform'][0]))
    assert peak1.item() == pytest.approx(target_peak, abs=1e-6)
    assert original_peak_levels[0].item() == pytest.approx(0.5, abs=1e-6)

    # Check item 2
    peak2 = torch.max(torch.abs(normalized_audio['waveform'][1]))
    assert peak2.item() == pytest.approx(target_peak, abs=1e-6)
    assert original_peak_levels[1].item() == pytest.approx(0.1, abs=1e-6)

    assert original_peak_levels.shape == (2,)


def test_normalize_silent_audio(normalize_node):
    """Tests that silent audio remains silent and doesn't cause errors."""
    sr = 44100
    silent_audio = {"waveform": torch.zeros(1, 1, sr), "sample_rate": sr}

    (normalized_audio, original_peak_level) = normalize_node.normalize_audio(silent_audio, 1.0)

    # Assert that the output is still silent
    assert torch.all(normalized_audio['waveform'] == 0)
    # The original peak level should be 0
    assert original_peak_level.item() == 0.0


# --- Tests for AudioApplyBatchedGain ---

def test_apply_gain_and_revert_normalization(normalize_node, apply_gain_node):
    """
    An integration test to verify that applying the `original_peak_level`
    from the normalize node correctly reverts the audio to its original state.
    """
    sr = 44100
    audio_orig = create_dummy_audio(3, 2, sr, sr, peak_value=0.3)

    # 1. Normalize the audio
    (normalized_audio, original_peak_levels) = normalize_node.normalize_audio(audio_orig, 1.0)

    # 2. Apply the original peak levels as gain to the normalized audio
    (reverted_audio,) = apply_gain_node.apply_gain(normalized_audio, original_peak_levels)

    # Assertions
    # The reverted audio should be almost identical to the original audio
    assert reverted_audio['sample_rate'] == audio_orig['sample_rate']
    assert reverted_audio['waveform'].shape == audio_orig['waveform'].shape
    # Check that the content is the same (with a small tolerance for float math)
    assert torch.allclose(reverted_audio['waveform'], audio_orig['waveform'], atol=1e-6)


def test_apply_gain_mismatched_batch_size_error(apply_gain_node):
    """
    Tests that an error is raised if the audio batch size and gain values
    batch size do not match.
    """
    sr = 44100
    audio = create_dummy_audio(3, 1, sr, sr)
    # Gain values for only 2 items, but audio has 3
    gain_values = torch.tensor([0.5, 0.5])

    with pytest.raises(ValueError, match="Batch size of audio .* and gain_values .* must match"):
        apply_gain_node.apply_gain(audio, gain_values)
