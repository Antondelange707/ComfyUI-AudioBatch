"""
Functional and regression tests for the AudioTestSignalGenerator node.
"""

import bootstrap  # noqa: F401
import logging
import pytest
import torch
import torchaudio.transforms as T

# Import the class and utility function to be tested
from nodes.nodes_audio import AudioTestSignalGenerator, logger as node_logger
# from utils.misc import parse_time_to_seconds

# Configure logging for tests to see output from the node
node_logger.setLevel(logging.DEBUG)


# --- Pytest Fixtures (Reusable Setup) ---

@pytest.fixture
def signal_gen_node():
    """Provides an instance of the AudioTestSignalGenerator node for tests."""
    return AudioTestSignalGenerator()


# --- Helper Functions for Strict Testing ---

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
    freq_bins = torch.fft.rfftfreq(n=waveform.size(-1), d=1.0 / sample_rate)
    # Find the index of the maximum magnitude in the FFT
    peak_index = torch.argmax(torch.abs(fft_result))
    # Get the frequency corresponding to that peak index
    peak_freq = freq_bins[peak_index].item()
    return float(peak_freq)


# --- Test Cases ---

def test_generator_basic_sine_shape_and_type(signal_gen_node):
    """Tests basic generation of a sine wave, checking output structure."""
    (result_audio,) = signal_gen_node.generate_signal(
        waveform_type="sine", frequency=440.0, frequency_end=880.0,
        amplitude=0.8, dc_offset=0.0, phase=0.0, duration="2.0",
        sample_rate=44100, batch_size=2, channels=2, seed=0
    )

    assert isinstance(result_audio, dict)
    assert "waveform" in result_audio
    assert "sample_rate" in result_audio

    waveform = result_audio['waveform']
    sr = result_audio['sample_rate']

    assert isinstance(waveform, torch.Tensor)
    assert sr == 44100
    expected_samples = int(2.0 * sr)
    assert waveform.shape == (2, 2, expected_samples)  # B, C, N
    assert waveform.dtype == torch.float32


@pytest.mark.parametrize("duration_str, expected_seconds", [
    ("5", 5.0),
    ("2.5", 2.5),
    ("0:10", 10.0),
    ("0:01.5", 1.5),
    ("1:02:03", 3723.0)
])
def test_generator_duration_parsing(signal_gen_node, duration_str, expected_seconds):
    """Tests if the duration string is parsed correctly to produce the right length."""
    sr = 16000
    (result_audio,) = signal_gen_node.generate_signal(
        waveform_type="sine", frequency=100.0, frequency_end=200.0,
        amplitude=1.0, dc_offset=0.0, phase=0.0, duration=duration_str,
        sample_rate=sr, batch_size=1, channels=1, seed=0
    )

    expected_samples = int(expected_seconds * sr)
    actual_samples = result_audio['waveform'].shape[2]
    assert actual_samples == expected_samples


def test_generator_fft_verification_sine(signal_gen_node):
    """
    Strict Test: Verifies that a generated sine wave has the correct peak frequency.
    """
    sr = 44100
    target_freq = 1000.0

    (result_audio,) = signal_gen_node.generate_signal(
        waveform_type="sine", frequency=target_freq, frequency_end=0.0,
        amplitude=0.9, dc_offset=0.0, phase=0.0, duration="1.0",
        sample_rate=sr, batch_size=1, channels=1, seed=0
    )

    peak_freq = get_peak_frequency(result_audio['waveform'], result_audio['sample_rate'])

    print(f"Target Frequency: {target_freq} Hz, Detected Peak Frequency: {peak_freq:.2f} Hz")
    # Assert that the detected frequency is very close to the target.
    # The tolerance accounts for the resolution of the FFT (bin width).
    assert peak_freq == pytest.approx(target_freq, abs=1.0)


def test_generator_amplitude_and_offset(signal_gen_node):
    """
    Strict Test: Verifies that amplitude and DC offset are correctly applied.
    """
    amplitude = 0.7
    offset = -0.1

    (result_audio,) = signal_gen_node.generate_signal(
        waveform_type="sine", frequency=100.0, frequency_end=0.0,
        amplitude=amplitude, dc_offset=offset, phase=90.0,  # Phase 90 ensures we hit the max value at t=0
        duration="1.0", sample_rate=44100, batch_size=1, channels=1, seed=0
    )

    waveform = result_audio['waveform']

    expected_max = amplitude + offset
    expected_min = -amplitude + offset

    # Check that the max and min values of the waveform are close to expected
    assert torch.max(waveform).item() == pytest.approx(expected_max, abs=1e-5)
    assert torch.min(waveform).item() == pytest.approx(expected_min, abs=1e-5)


def test_generator_noise_reproducibility_with_seed(signal_gen_node):
    """
    Strict Test: Verifies that using the same seed produces identical noise,
    and different seeds produce different noise.
    """
    params = {
        "waveform_type": "white_noise", "frequency": 0.0, "frequency_end": 0.0,
        "amplitude": 1.0, "dc_offset": 0.0, "phase": 0.0, "duration": "0.5",
        "sample_rate": 16000, "batch_size": 1, "channels": 1
    }

    # Generate twice with the same seed
    (result1,) = signal_gen_node.generate_signal(**params, seed=123)
    (result2,) = signal_gen_node.generate_signal(**params, seed=123)

    # Generate once with a different seed
    (result3,) = signal_gen_node.generate_signal(**params, seed=456)

    # Assertions
    assert torch.equal(result1['waveform'], result2['waveform'])
    assert not torch.equal(result1['waveform'], result3['waveform'])


def test_generator_impulse_content(signal_gen_node):
    """
    Strict Test: Verifies that an impulse waveform is correct.
    """
    amplitude = 0.9
    offset = 0.05

    (result_audio,) = signal_gen_node.generate_signal(
        waveform_type="impulse", frequency=0.0, frequency_end=0.0,
        amplitude=amplitude, dc_offset=offset, phase=0.0, duration="0.1",
        sample_rate=100, batch_size=1, channels=1, seed=0
    )

    waveform = result_audio['waveform'].squeeze()  # Get 1D tensor

    # The first sample should be the impulse
    expected_impulse_value = amplitude + offset
    assert waveform[0].item() == pytest.approx(expected_impulse_value)

    # All other samples should be just the DC offset
    assert torch.all(torch.isclose(waveform[1:], torch.tensor(offset)))


def linear_regression(x, y):
    """
    Performs a simple linear regression and returns the slope and intercept.
    Uses torch for calculations.
    """
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    numerator = torch.sum((x - x_mean) * (y - y_mean))
    denominator = torch.sum((x - x_mean) ** 2)
    if denominator == 0:
        return 0.0, y_mean.item()  # No slope if x is constant
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope.item(), intercept.item()


def test_generator_sweep_frequency_trend(signal_gen_node):
    """
    Strict Test (Corrected Again): Verifies a linear sweep has a constant rate of
    frequency change using STFT and linear regression.
    """
    sr = 44100
    f_start = 500.0
    f_end = 8000.0
    duration_seconds = 2.0

    (result_audio,) = signal_gen_node.generate_signal(
        waveform_type="sweep", frequency=f_start, frequency_end=f_end,
        amplitude=1.0, dc_offset=0.0, phase=0.0, duration=str(duration_seconds),
        sample_rate=sr, batch_size=1, channels=1, seed=0
    )

    waveform = result_audio['waveform'].squeeze()

    # --- Use STFT to analyze the frequency over time ---
    n_fft = 2048
    hop_length = 512

    spectrogram_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)
    spectrogram = spectrogram_transform(waveform)  # (freq_bins, time_frames)

    # Get the frequency and time axes for the spectrogram
    fft_freqs = torch.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    time_bins = torch.linspace(0, duration_seconds, steps=spectrogram.shape[1])

    # For each time slice, find the index of the peak frequency
    peak_freq_indices = torch.argmax(torch.abs(spectrogram), dim=0)

    # Convert these indices to actual frequencies
    detected_peak_freqs = fft_freqs[peak_freq_indices]

    # We might get some noise at the very beginning/end, so let's analyze the stable middle part
    start_idx_for_fit = 10
    end_idx_for_fit = -10

    if len(time_bins) <= start_idx_for_fit * 2:  # Handle very short signals
        start_idx_for_fit = 0
        end_idx_for_fit = len(time_bins)

    time_for_fit = time_bins[start_idx_for_fit:end_idx_for_fit]
    freqs_for_fit = detected_peak_freqs[start_idx_for_fit:end_idx_for_fit]

    # --- Perform Linear Regression ---
    # We are fitting a line: frequency = slope * time + intercept
    detected_slope, detected_intercept = linear_regression(time_for_fit, freqs_for_fit)

    # --- Calculate Theoretical Values ---
    theoretical_slope = (f_end - f_start) / duration_seconds
    theoretical_intercept = f_start

    logging.info("\n--- Sweep Linear Regression Test ---")
    logging.info(f"Theoretical Slope: {theoretical_slope:.2f} Hz/s")
    logging.info(f"Detected Slope:    {detected_slope:.2f} Hz/s")
    logging.info(f"Theoretical Intercept (Start Freq): {theoretical_intercept:.2f} Hz")
    logging.info(f"Detected Intercept:                 {detected_intercept:.2f} Hz")

    # Assertions:
    # 1. The detected slope should be very close to the theoretical rate of change.
    assert detected_slope == pytest.approx(theoretical_slope, rel=0.05)  # 5% tolerance is reasonable

    # 2. The detected intercept should be close to the starting frequency.
    #    This is still subject to some error from the first few windows, so give it some tolerance.
    assert detected_intercept == pytest.approx(theoretical_intercept, rel=0.1)  # 10% tolerance


def test_generator_silence_and_dc_offset(signal_gen_node):
    """
    Strict Test: Verifies the 'silence' waveform type and that it correctly
    applies the dc_offset.
    """
    sample_rate = 1000  # Use a low SR for a small tensor
    duration = "0.5"  # 0.5 seconds
    num_samples = int(0.5 * sample_rate)
    target_offset = 0.25

    # Generate a "silence" waveform with a non-zero DC offset.
    # Amplitude should have no effect.
    (result_audio,) = signal_gen_node.generate_signal(
        waveform_type="silence",
        frequency=100.0,  # Should be ignored
        frequency_end=800.0,  # Should be ignored
        amplitude=0.8,   # Should be ignored
        dc_offset=target_offset,
        phase=0.0,
        duration=duration,
        sample_rate=sample_rate,
        batch_size=2,
        channels=2,
        seed=0
    )

    waveform = result_audio['waveform']

    # 1. Check shape and SR
    assert waveform.shape == (2, 2, num_samples)
    assert result_audio['sample_rate'] == sample_rate

    # 2. Check content
    # The entire tensor should be filled with the dc_offset value.
    expected_waveform = torch.full_like(waveform, fill_value=target_offset)

    assert torch.allclose(waveform, expected_waveform)

    # 3. Test pure silence (dc_offset = 0)
    (result_pure_silence,) = signal_gen_node.generate_signal(
        waveform_type="silence",
        dc_offset=0.0,
        # ... other params ...
        frequency=100.0, frequency_end=800.0, amplitude=0.8, phase=0.0, duration=duration,
        sample_rate=sample_rate, batch_size=1, channels=1, seed=0
    )

    assert torch.all(result_pure_silence['waveform'] == 0.0)
