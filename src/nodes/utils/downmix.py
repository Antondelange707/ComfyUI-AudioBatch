# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: GPLv3
# Project: ComfyUI-AudioBatch
#
# Audio batch aligner
# Original code from Gemini 2.5 Pro
import logging
import torch
from .. import NODES_NAME

logger = logging.getLogger(f"{NODES_NAME}.DownmixUtil")


# --- Using Demucs robust STFT and iSTFT implementations ---
# These are excellent as they handle windowing, device fallbacks, and reshaping.

def spectro(x: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    """
    Computes the Short-Time Fourier Transform (STFT) of a tensor.
    Handles batching and MPS device fallbacks.
    """
    *other, length = x.shape
    x = x.reshape(-1, length)  # Flatten batch and channel dimensions

    # Handle MPS device which might not support STFT with all options
    is_mps = x.device.type == 'mps'
    original_device = x.device
    if is_mps:
        x = x.cpu()

    window = torch.hann_window(n_fft, device=x.device)

    z = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        pad_mode='reflect',
        normalized=True,
        return_complex=True
    )

    if is_mps:
        z = z.to(original_device)  # Move result back to original device

    _, freqs, frames = z.shape
    return z.view(*other, freqs, frames)


def ispectro(z: torch.Tensor, hop_length: int, length: int) -> torch.Tensor:
    """
    Computes the Inverse Short-Time Fourier Transform (iSTFT) of a complex tensor.
    """
    *other, freqs, frames = z.shape
    n_fft = 2 * (freqs - 1)
    z = z.view(-1, freqs, frames)

    is_mps = z.device.type == 'mps'
    original_device = z.device
    if is_mps:
        z = z.cpu()

    window = torch.hann_window(n_fft, device=z.device)

    x = torch.istft(
        z,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        normalized=True,
        length=length
    )

    if is_mps:
        x = x.to(original_device)

    _, out_length = x.shape
    return x.view(*other, out_length)


def spectral_downmix(
    waveform: torch.Tensor,  # (B, C, N) stereo or multi-channel
    n_fft: int = 2048,
    hop_length: int = 512
) -> torch.Tensor:  # Returns (B, 1, N) mono
    """
    Downmixes a stereo or multi-channel audio to mono in the spectral domain
    to avoid phase cancellation, using robust STFT/iSTFT functions.
    """
    if waveform.shape[1] <= 1:
        logger.debug("Input is already mono, skipping spectral downmix.")
        return waveform

    batch_size, num_channels, num_samples = waveform.shape

    # 1. Perform STFT on the multi-channel waveform
    # The spectro function handles reshaping (B,C,N) -> (B*C,N) if needed,
    # but passing it directly as (B,C,N) is also fine as it correctly reshapes.
    spectrogram_multi_ch = spectro(waveform, n_fft=n_fft, hop_length=hop_length)
    # Shape is (B, C, F, T_spec)

    # 2. Separate magnitude and phase
    magnitudes = spectrogram_multi_ch.abs()
    phases = spectrogram_multi_ch.angle()

    # 3. Average the magnitudes across channels
    avg_magnitude = torch.mean(magnitudes, dim=1)  # (B, F, T_spec)

    # 4. Use the phase from the first channel as the reference
    ref_phase = phases[:, 0, :, :]  # (B, F, T_spec)

    # 5. Reconstruct the new mono complex spectrogram
    mono_spectrogram_complex = torch.polar(avg_magnitude, ref_phase)

    # 6. Perform inverse STFT
    # ispectro expects (..., F, T_spec) and returns (..., N)
    # Our mono_spectrogram_complex is (B, F, T_spec), which is perfect.
    mono_waveform_time = ispectro(mono_spectrogram_complex, hop_length=hop_length, length=num_samples)

    # Reshape to (B, 1, N) for ComfyUI AUDIO standard
    return mono_waveform_time.unsqueeze(1)
