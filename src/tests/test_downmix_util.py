"""
Functional and regression tests for the AudioTestSignalGenerator node.
"""

import bootstrap  # noqa: F401
import torch
import pytest
from nodes.utils.downmix import spectral_downmix


# Helper to calculate Signal-to-Noise Ratio (SNR)
def calculate_snr(signal, noisy_signal):
    noise = noisy_signal - signal
    signal_power = torch.mean(signal ** 2)
    noise_power = torch.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * torch.log10(signal_power / noise_power).item()


def test_spectral_downmix_preserves_in_phase_signal():
    """
    Tests that an in-phase stereo signal (L=R) is preserved correctly,
    measuring the SNR of the output vs the original mono source.
    """
    sr = 44100
    samples = sr

    # Create a mono sine wave
    t = torch.linspace(0., 1., samples)
    mono_signal = torch.sin(2 * torch.pi * 440.0 * t)

    # Create a perfect stereo signal where L=R=mono_signal
    stereo_waveform = mono_signal.unsqueeze(0).repeat(1, 2, 1)  # (1, 2, N)

    # Run spectral downmix
    mono_output = spectral_downmix(stereo_waveform).squeeze()  # Get 1D tensor

    # The output should be very close to the original mono signal
    snr = calculate_snr(mono_signal, mono_output)

    print(f"SNR for in-phase signal: {snr:.2f} dB")
    # A high SNR indicates the signals are very similar. >30dB is very good for this.
    assert snr > 30.0


def test_spectral_downmix_avoids_antiphase_cancellation():
    """
    Strict Test: Verifies that spectral downmix AVOIDS phase cancellation
    for anti-phase signals, which is its main advantage.
    """
    sr = 44100
    samples = sr

    t = torch.linspace(0., 1., samples)
    left_channel = torch.sin(2 * torch.pi * 440.0 * t)
    right_channel = -left_channel  # Perfectly anti-phase

    stereo_waveform = torch.stack((left_channel, right_channel), dim=0).unsqueeze(0)

    # Run spectral downmix
    mono_output_spectral = spectral_downmix(stereo_waveform).squeeze()

    # The magnitudes are identical, so the output should have the same magnitude
    # as one of the original channels. We check the power (mean of squares).
    output_power = torch.mean(mono_output_spectral ** 2)
    original_power = torch.mean(left_channel ** 2)

    print(f"Power of original L channel: {original_power:.4f}")
    print(f"Power of spectral downmix output: {output_power:.4f}")

    # The output power should be very close to the original channel's power.
    # A simple time-domain average would result in near-zero power.
    assert output_power > original_power * 0.9  # Should be very close
    assert output_power != pytest.approx(0.0)
