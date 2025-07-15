"""
Regression tests for the AudioBatch node in ComfyUI-AudioBatch.
"""

import bootstrap  # noqa: F401
import torch
import pytest
from nodes.nodes_audio import AudioBatch


# Helper function (can be moved to a shared conftest.py or test_utils.py later)
def create_dummy_audio(batch_size, channels, samples, sr, device='cpu'):
    waveform = torch.randn(batch_size, channels, samples, device=device, dtype=torch.float32)
    return {"waveform": waveform, "sample_rate": sr}


@pytest.fixture
def audio_batch_node():
    return AudioBatch()


def test_batch_simple_match(audio_batch_node):
    """Tests batching two identical audio inputs."""
    sr = 44100
    audio1 = create_dummy_audio(1, 2, 1000, sr)
    audio2 = create_dummy_audio(1, 2, 1000, sr)

    (result_audio,) = audio_batch_node.batch_audio(audio1, audio2)

    expected_waveform = torch.cat((audio1['waveform'], audio2['waveform']), dim=0)

    assert result_audio['sample_rate'] == sr
    assert result_audio['waveform'].shape == (2, 2, 1000)
    assert torch.allclose(result_audio['waveform'], expected_waveform)


def test_batch_different_batch_sizes(audio_batch_node):
    """Tests batching inputs with B=2 and B=3, expecting an output of B=5."""
    sr = 44100
    audio1 = create_dummy_audio(2, 2, 1000, sr)
    audio2 = create_dummy_audio(3, 2, 1000, sr)

    (result_audio,) = audio_batch_node.batch_audio(audio1, audio2)

    assert result_audio['waveform'].shape[0] == 5  # 2 + 3


def test_batch_different_lengths_padding(audio_batch_node):
    """Tests if the shorter audio is padded to match the longer one."""
    sr = 44100
    audio1 = create_dummy_audio(1, 2, 500, sr)
    audio2 = create_dummy_audio(1, 2, 1000, sr)

    (result_audio,) = audio_batch_node.batch_audio(audio1, audio2)

    # Both items in the output batch should have the length of the longer audio
    assert result_audio['waveform'].shape[2] == 1000
    # The first item should have been padded
    assert result_audio['waveform'][0].shape[1] == 1000


def test_batch_mono_and_stereo(audio_batch_node):
    """Tests if the mono audio is converted to stereo to match the other input."""
    sr = 44100
    audio1_mono = create_dummy_audio(1, 1, 1000, sr)
    audio2_stereo = create_dummy_audio(1, 2, 1000, sr)

    (result_audio,) = audio_batch_node.batch_audio(audio1_mono, audio2_stereo)

    # Both items in the output batch should be stereo
    assert result_audio['waveform'].shape[1] == 2
    assert result_audio['waveform'].shape[0] == 2


def test_batch_different_sample_rates(audio_batch_node):
    """Tests if audio2 is resampled to match audio1's sample rate."""
    sr1, sr2 = 44100, 22050
    samples1, samples2 = 2000, 1000
    audio1 = create_dummy_audio(1, 2, samples1, sr1)
    audio2 = create_dummy_audio(1, 2, samples2, sr2)

    (result_audio,) = audio_batch_node.batch_audio(audio1, audio2)

    # Output SR should be sr1
    assert result_audio['sample_rate'] == sr1
    # Length of audio2 after resampling is approx samples2 * (sr1/sr2) = 1000 * 2 = 2000
    # So both items should have length 2000
    assert result_audio['waveform'].shape[2] == 2000


def test_batch_full_alignment_and_content(audio_batch_node):
    """
    Strictly tests the end-to-end alignment (SR, channels, length) and content
    of the batching process.
    """
    # --- Input 1: A stereo signal at 48000 Hz, 1 second long ---
    sr1 = 48000
    len1 = sr1 * 1
    # Create L/R channels that are easy to identify (ramps)
    wf1_l = torch.linspace(0.1, 0.2, len1)
    wf1_r = torch.linspace(0.3, 0.4, len1)
    wf1_orig = torch.stack((wf1_l, wf1_r), dim=0).unsqueeze(0)  # (1, 2, 48000)
    audio1 = {"waveform": wf1_orig, "sample_rate": sr1}

    # --- Input 2: A mono signal at 16000 Hz, 1.5 seconds long ---
    sr2 = 16000
    len2 = int(sr2 * 1.5)
    wf2_orig = (torch.ones(1, 1, len2) * 0.5)  # Constant value mono signal, (1, 1, 24000)
    audio2 = {"waveform": wf2_orig, "sample_rate": sr2}

    # --- Action ---
    (result_audio,) = audio_batch_node.batch_audio(audio1, audio2)
    result_wf = result_audio['waveform']

    # --- Assertions ---
    # 1. Final parameters should match audio1's SR, be stereo, and have the longest length after resampling.
    target_sr = sr1  # 48000
    target_channels = 2
    len2_resampled = int(len2 * (target_sr / sr2))  # 24000 * 3 = 72000
    target_samples = max(len1, len2_resampled)  # max(48000, 72000) = 72000

    assert result_audio['sample_rate'] == target_sr
    assert result_wf.shape == (2, target_channels, target_samples)  # B=2, C=2, N=72000

    # 2. Verify content of the first item (was audio1)
    item1_output = result_wf[0]  # (2, 72000)
    # The first part should match the original audio1
    assert torch.allclose(item1_output[:, :len1], wf1_orig.squeeze(0))
    # The padded part should be zeros
    assert torch.all(item1_output[:, len1:] == 0)

    # 3. Verify content of the second item (was audio2)
    item2_output = result_wf[1]  # (2, 72000)
    # It was mono, so both output channels should be identical
    assert torch.allclose(item2_output[0, :], item2_output[1, :])
    # The content should be a resampled version of the constant 0.5 signal.
    # A resampled constant signal should still be (roughly) constant.
    # We check the first part, up to where the original content was.
    resampled_part = item2_output[0, :len2_resampled]
    # The average value should be very close to 0.5. Due to filtering (Gibbs effect),
    # it won't be perfect, especially at the edges.
    assert torch.mean(resampled_part[100:-100]).item() == pytest.approx(0.5, abs=1e-3)
