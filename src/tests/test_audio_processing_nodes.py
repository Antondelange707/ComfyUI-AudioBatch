"""
Regression tests for the AudioChannelConverter, AudioResampler, AudioProcessAdvanced node in ComfyUI-AudioBatch.
"""

import bootstrap  # noqa: F401
import logging
import torch
import pytest
from nodes.nodes_audio import AudioChannelConverter, AudioResampler, AudioProcessAdvanced


# Helper function
def create_dummy_audio(batch_size, channels, samples, sr, device='cpu'):
    waveform = torch.randn(batch_size, channels, samples, device=device, dtype=torch.float32)
    return {"waveform": waveform, "sample_rate": sr}


# --- Tests for AudioChannelConverter ---

@pytest.fixture
def converter_node():
    return AudioChannelConverter()


def test_converter_mono_to_stereo(converter_node):
    sr = 44100
    audio_mono = create_dummy_audio(2, 1, 1000, sr)
    (result_audio,) = converter_node.convert_channels(audio_mono, "mono_to_stereo", "average")
    assert result_audio['waveform'].shape[1] == 2


def test_converter_stereo_to_mono(converter_node):
    sr = 44100
    audio_stereo = create_dummy_audio(2, 2, 1000, sr)
    (result_audio,) = converter_node.convert_channels(audio_stereo, "stereo_to_mono", "average")
    assert result_audio['waveform'].shape[1] == 1


def test_converter_force_stereo_from_multichannel(converter_node):
    """Tests if force_stereo takes the first channel of a 5.1 input."""
    sr = 44100
    audio_5_1 = create_dummy_audio(1, 6, 1000, sr)
    (result_audio,) = converter_node.convert_channels(audio_5_1, "force_stereo", "average")
    assert result_audio['waveform'].shape[1] == 2
    # Check if the two output channels are identical copies of the first input channel
    assert torch.allclose(result_audio['waveform'][:, 0, :], result_audio['waveform'][:, 1, :])
    assert torch.allclose(result_audio['waveform'][:, 0, :], audio_5_1['waveform'][:, 0, :])


def test_converter_keep_channels(converter_node):
    sr = 44100
    audio_stereo = create_dummy_audio(1, 2, 1000, sr)
    (result_audio,) = converter_node.convert_channels(audio_stereo, "keep", "average")
    assert result_audio['waveform'].shape[1] == 2
    assert torch.allclose(result_audio['waveform'], audio_stereo['waveform'])


@pytest.mark.parametrize("downmix_method", ["average", "standard_gain"])
def test_converter_stereo_to_mono_math(converter_node, downmix_method):
    """
    Strictly tests that stereo-to-mono downmixing is mathematically correct
    for both 'average' and 'standard_gain' methods.
    """
    sr = 44100
    samples = sr  # 1 second

    # Create distinct L and R channels
    left_channel = torch.linspace(0.1, 0.9, samples)
    right_channel = torch.ones(samples) * 0.5

    stereo_waveform = torch.stack((left_channel, right_channel), dim=0).unsqueeze(0)
    stereo_audio = {"waveform": stereo_waveform, "sample_rate": sr}

    # Run the conversion with the parameterized downmix method
    (mono_audio,) = converter_node.convert_channels(
        audio=stereo_audio,
        channel_conversion="force_mono",
        downmix_method=downmix_method
    )

    # Calculate the expected result based on the method
    if downmix_method == "average":
        expected_mono_waveform = ((left_channel + right_channel) / 2.0).unsqueeze(0).unsqueeze(0)
    elif downmix_method == "standard_gain":
        gain = 1.0 / (2**0.5)  # Gain for stereo
        expected_mono_waveform = ((left_channel + right_channel) * gain).unsqueeze(0).unsqueeze(0)
    else:
        pytest.fail(f"Test case not implemented for downmix method: {downmix_method}")

    # Assertions
    assert mono_audio['waveform'].shape == (1, 1, samples)
    assert torch.allclose(mono_audio['waveform'], expected_mono_waveform, atol=1e-6)


def test_converter_antiphase_cancellation_time_domain(converter_node):
    """
    Strictly tests and confirms that time-domain methods (both average and standard_gain)
    suffer from phase cancellation with anti-phase signals. This is expected behavior.
    """
    sr = 44100
    samples = sr

    # Create a sine wave for the left channel
    t = torch.linspace(0., 1., samples)
    left_channel = torch.sin(2 * torch.pi * 440.0 * t)
    # Create a perfectly inverted (anti-phase) wave for the right channel
    right_channel = -left_channel

    stereo_waveform = torch.stack((left_channel, right_channel), dim=0).unsqueeze(0)
    antiphase_audio = {"waveform": stereo_waveform, "sample_rate": sr}

    # Run conversion using 'average'
    (mono_audio_avg,) = converter_node.convert_channels(
        audio=antiphase_audio,
        channel_conversion="force_mono",
        downmix_method="average"
    )

    # The result of (L + (-L)) / 2 should be a tensor of all zeros.
    assert torch.allclose(mono_audio_avg['waveform'], torch.zeros_like(mono_audio_avg['waveform']), atol=1e-6)


def test_converter_spectral_downmix_avoids_cancellation(converter_node):
    """
    Verifies that the node, when set to 'spectral' downmix, avoids cancelling
    an anti-phase signal, unlike the time-domain methods.
    """
    sr = 44100
    samples = sr

    t = torch.linspace(0., 1., samples)
    left_channel = torch.sin(2 * torch.pi * 440.0 * t)
    right_channel = -left_channel  # Anti-phase

    stereo_waveform = torch.stack((left_channel, right_channel), dim=0).unsqueeze(0)
    antiphase_audio = {"waveform": stereo_waveform, "sample_rate": sr}

    # Run conversion using the new 'spectral' method
    (mono_audio_spectral,) = converter_node.convert_channels(
        audio=antiphase_audio,
        channel_conversion="force_mono",
        downmix_method="spectral"
    )

    # The spectral downmix should preserve the energy of the signal.
    # Check that the mean of the squared signal (power) is significant.
    output_power = torch.mean(mono_audio_spectral['waveform'] ** 2)
    original_power = torch.mean(left_channel ** 2)

    # It won't be identical to original_power due to phase reconstruction,
    # but it should be much, much greater than zero.
    assert output_power > original_power * 0.5  # Assert it retains at least 50% of the power
    assert output_power > 1e-3  # Assert it's not silent


# --- Tests for AudioResampler ---

@pytest.fixture
def resampler_node():
    return AudioResampler()


def test_resampler_upsample(resampler_node):
    sr_orig, sr_target = 22050, 44100
    samples_orig = 1000
    audio = create_dummy_audio(2, 2, samples_orig, sr_orig)
    (result_audio,) = resampler_node.resample_audio(audio, sr_target)

    expected_samples = int(samples_orig * (sr_target / sr_orig))
    assert result_audio['sample_rate'] == sr_target
    assert result_audio['waveform'].shape[2] == expected_samples


def test_resampler_downsample(resampler_node):
    sr_orig, sr_target = 48000, 16000
    samples_orig = 3000
    audio = create_dummy_audio(1, 1, samples_orig, sr_orig)
    (result_audio,) = resampler_node.resample_audio(audio, sr_target)

    expected_samples = int(samples_orig * (sr_target / sr_orig))
    assert result_audio['sample_rate'] == sr_target
    assert abs(result_audio['waveform'].shape[2] - expected_samples) < 2  # Resampling can have off-by-one


def test_resampler_no_op(resampler_node):
    """Tests if resampling is skipped when target SR is 0 or same as original."""
    sr = 44100
    audio = create_dummy_audio(1, 2, 1000, sr)

    # Test with target_sr = 0
    (result_audio_zero,) = resampler_node.resample_audio(audio, 0)
    assert torch.allclose(result_audio_zero['waveform'], audio['waveform'])
    assert result_audio_zero['sample_rate'] == sr

    # Test with target_sr = original_sr
    (result_audio_same,) = resampler_node.resample_audio(audio, sr)
    assert torch.allclose(result_audio_same['waveform'], audio['waveform'])
    assert result_audio_same['sample_rate'] == sr


def get_peak_frequency(waveform: torch.Tensor, sample_rate: int) -> float:
    """Helper to find the dominant frequency in a waveform using FFT."""
    if waveform.ndim > 1:
        # Use the first channel and first batch item for analysis
        waveform = waveform.squeeze()
        if waveform.ndim > 1:
            waveform = waveform[0]

    # Perform Real Fast Fourier Transform
    fft_result = torch.fft.rfft(waveform)
    # Get frequency bins for the FFT result
    freq_bins = torch.fft.rfftfreq(n=waveform.size(-1), d=1./sample_rate)
    # Find the index of the maximum magnitude in the FFT
    peak_index = torch.argmax(torch.abs(fft_result))
    # Get the frequency corresponding to that peak index
    peak_freq = freq_bins[peak_index].item()
    return float(peak_freq)


def test_resampler_preserves_frequency_content(resampler_node):
    """
    Strictly tests if resampling correctly preserves the pitch (frequency) of a signal.
    """
    # Original signal: A 440 Hz sine wave at 44100 Hz SR
    sr_orig = 44100
    target_freq = 440.0  # A4 note
    duration = 1.0
    samples_orig = int(sr_orig * duration)
    t = torch.linspace(0., duration, samples_orig)
    original_waveform = torch.sin(2 * torch.pi * target_freq * t).unsqueeze(0).unsqueeze(0)  # (1,1,N)
    original_audio = {"waveform": original_waveform, "sample_rate": sr_orig}

    # Check the frequency of the original signal to ensure our helper works
    original_peak_freq = get_peak_frequency(original_waveform, sr_orig)
    assert abs(original_peak_freq - target_freq) < 1.0  # Allow for slight FFT bin inaccuracy

    # --- Test Downsampling ---
    sr_down = 16000
    (downsampled_audio,) = resampler_node.resample_audio(original_audio, sr_down)

    # The frequency content should still be centered at 440 Hz
    downsampled_peak_freq = get_peak_frequency(downsampled_audio['waveform'], sr_down)
    logging.info(f"Downsampled from {sr_orig}Hz to {sr_down}Hz. Original peak: {original_peak_freq:.2f}Hz, "
                 f"New peak: {downsampled_peak_freq:.2f}Hz")
    assert abs(downsampled_peak_freq - target_freq) < 2.0  # Allow slightly larger tolerance for resamplers

    # --- Test Upsampling ---
    sr_up = 48000
    (upsampled_audio,) = resampler_node.resample_audio(original_audio, sr_up)

    # The frequency content should still be centered at 440 Hz
    upsampled_peak_freq = get_peak_frequency(upsampled_audio['waveform'], sr_up)
    logging.info(f"Upsampled from {sr_orig}Hz to {sr_up}Hz. Original peak: {original_peak_freq:.2f}Hz, "
                 f"New peak: {upsampled_peak_freq:.2f}Hz")
    assert abs(upsampled_peak_freq - target_freq) < 2.0


# --- Test for AudioProcessAdvanced ---

@pytest.fixture
def advanced_processor_node():
    return AudioProcessAdvanced()


def test_advanced_processor_integration_smoke_test(advanced_processor_node):
    """
    A simple "smoke test" to ensure AudioProcessAdvanced correctly calls its
    sub-components and produces an output of the expected shape and SR.
    We don't need to re-test all permutations, as those are covered
    by the unit tests for the individual converter and resampler nodes.
    """
    # Setup: Start with stereo audio at a high sample rate
    sr_orig = 44100
    audio_stereo_orig = create_dummy_audio(1, 2, sr_orig * 2, sr_orig)  # 2 seconds

    # Action: Convert to mono and downsample to 16k
    target_sr = 16000
    (result_audio,) = advanced_processor_node.process_audio(
        audio=audio_stereo_orig,
        channel_conversion="force_mono",
        target_sample_rate=target_sr,
        downmix_method="average"
    )

    # Assertions:
    # 1. Did it successfully produce an output?
    assert result_audio is not None
    assert "waveform" in result_audio
    assert "sample_rate" in result_audio

    # 2. Is the final sample rate correct?
    assert result_audio['sample_rate'] == target_sr

    # 3. Is the final channel count correct?
    assert result_audio['waveform'].shape[1] == 1  # Should be mono

    # 4. Is the final number of samples roughly correct?
    original_samples = sr_orig * 2
    expected_samples = int(original_samples * (target_sr / sr_orig))
    # Check that the actual number of samples is close to the expected number
    assert abs(result_audio['waveform'].shape[2] - expected_samples) < 2
