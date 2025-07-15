# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: GPLv3
# Project: ComfyUI-AudioBatch
#
# Audio batch aligner
# Original code from Gemini 2.5 Pro
import logging
import torch
import torchaudio.transforms as T
from .misc import NODES_NAME

logger = logging.getLogger(f"{NODES_NAME}.aligner")


def convert_batch_to_stereo_tensor(audio_waveform_mono_batch: torch.Tensor) -> torch.Tensor:
    """ Converts a batch of mono audio tensors (B, 1, N) to stereo (B, 2, N). """
    if audio_waveform_mono_batch.ndim != 3 or audio_waveform_mono_batch.shape[1] != 1:
        # This could also happen if an input was (N) or (1,N) and wasn't unsqueezed to (B,1,N) yet
        raise ValueError("Input for stereo conversion must be a batch of mono audio (B, 1, N), "
                         f"got {audio_waveform_mono_batch.shape}")
    return audio_waveform_mono_batch.repeat(1, 2, 1)  # (B, 1, N) -> (B, 2, N)


class AudioBatchAligner:
    """ A helper class to align two audio batches for processing. """
    def __init__(self, audio1: dict, audio2: dict):
        self.waveform1_orig, self.sr1 = audio1['waveform'], audio1['sample_rate']
        self.waveform2_orig, self.sr2 = audio2['waveform'], audio2['sample_rate']

        logger.debug(f"Audio1 input: shape={self.waveform1_orig.shape}, sr={self.sr1}, dtype={self.waveform1_orig.dtype}, "
                     f"device={self.waveform1_orig.device}")
        logger.debug(f"Audio2 input: shape={self.waveform2_orig.shape}, sr={self.sr2}, dtype={self.waveform2_orig.dtype}, "
                     f"device={self.waveform2_orig.device}")

        # Use properties of audio1 as the reference for the output batch
        self.reference_device = self.waveform1_orig.device
        self.reference_dtype = self.waveform1_orig.dtype
        self.target_sr = self.sr1  # All audio will be resampled to sr1

        # This will be populated by _determine_target_params
        self.target_channels: int = 0
        self.target_samples: int = 0

        self._determine_target_params()

    def _determine_target_params(self):
        """ Determines the target channels and samples for the unified batch. """
        c1, c2 = self.waveform1_orig.shape[1], self.waveform2_orig.shape[1]

        if (c1 == 1 and c2 == 2) or (c1 == 2 and c2 == 1) or (c1 == 2 and c2 == 2):
            self.target_channels = 2
        elif c1 == 1 and c2 == 1:
            self.target_channels = 1
        else:
            # For other multi-channel counts (e.g. 5.1), this simple logic might not be ideal.
            # For now, default to max, but this could be an error or require downmixing node.
            logger.warning(f"Complex channel counts detected (C1={c1}, C2={c2}). Defaulting to max channels "
                           f"({max(c1,c2)}) and hoping for downstream compatibility. Explicit mono/stereo "
                           "alignment is preferred for this node.")
            self.target_channels = max(c1, c2)

        # Determine target number of samples (length) after potential resampling
        n1_orig, n2_orig = self.waveform1_orig.shape[2], self.waveform2_orig.shape[2]
        n1_after_resample = n1_orig  # No resampling for audio1 as it's the target_sr reference
        n2_after_resample = n2_orig if self.sr2 == self.target_sr else int(n2_orig * (self.target_sr / self.sr2))
        self.target_samples = max(n1_after_resample, n2_after_resample)

        logger.debug(f"Unified Batch Params: SR={self.target_sr}, Channels={self.target_channels}, "
                     f"Samples={self.target_samples}")

    def _align_one_batch(self, waveform: torch.Tensor, original_sr: int) -> torch.Tensor:
        """ Processes a single waveform batch to match the target parameters. """
        # Ensure correct device and dtype
        processed_wf = waveform.to(device=self.reference_device, dtype=self.reference_dtype)
        current_batch_size, current_channels, current_samples = processed_wf.shape

        # 1. Resample if necessary
        if original_sr != self.target_sr:
            logger.debug(f"Resampling batch from {original_sr} to {self.target_sr}. Input shape: {processed_wf.shape}")
            # Resample expects (..., time)
            # For (B, C, N), we can reshape to (B*C, N), resample, then reshape back.
            # Or, if T.Resample handles batch dims appropriately (some versions might if C=1 or applied per channel).
            # Let's reshape for robustness with T.Resample.
            resampler = T.Resample(orig_freq=original_sr, new_freq=self.target_sr, dtype=self.reference_dtype,
                                   lowpass_filter_width=24).to(self.reference_device)

            if current_channels == 1:
                # Reshape (B, 1, N) to (B, N) for resampler, then unsqueeze back
                processed_wf = resampler(processed_wf.squeeze(1)).unsqueeze(1)  # (B, 1, N_new)
            else:  # Multi-channel (e.g., stereo)
                # Resample each channel in the batch separately
                # This is more complex if T.Resample doesn't broadcast correctly.
                # A common way: permute to (C, B, N), reshape to (C*B, N), resample, reshape back.
                # Or loop (less efficient for GPU tensors).
                # Let's try reshaping to (B*C, N)
                original_shape = processed_wf.shape
                processed_wf_flat = processed_wf.reshape(-1, current_samples)  # (B*C, N)
                resampled_wf_flat = resampler(processed_wf_flat)  # (B*C, N_new)
                # Reshape back to (B, C, N_new)
                processed_wf = resampled_wf_flat.reshape(original_shape[0], original_shape[1], -1)

            current_samples = processed_wf.shape[2]  # Update sample count
            logger.debug(f"Batch after resampling: shape={processed_wf.shape}")

        # 2. Adjust number of channels
        if current_channels != self.target_channels:
            if current_channels == 1 and self.target_channels == 2:
                processed_wf = convert_batch_to_stereo_tensor(processed_wf)  # (B, 1, N) -> (B, 2, N)
                logger.debug(f"Batch converted to stereo: shape={processed_wf.shape}")
            elif current_channels == 2 and self.target_channels == 1:
                # Example: Convert stereo to mono by averaging (can make this an option)
                processed_wf = processed_wf.mean(dim=1, keepdim=True)  # (B, 2, N) -> (B, 1, N)
                logger.debug(f"Batch converted to mono (avg): shape={processed_wf.shape}")
            else:
                raise ValueError(f"Cannot align channels: input has {current_channels}, target is {self.target_channels}")

        # 3. Pad length if necessary
        if current_samples < self.target_samples:
            padding_needed = self.target_samples - current_samples
            # Pad only the last dimension (samples)
            processed_wf = torch.nn.functional.pad(processed_wf, (0, padding_needed))
            logger.debug(f"Batch padded: shape={processed_wf.shape}")
        elif current_samples > self.target_samples:  # Should not happen if target_samples is max length
            logger.warning(f"Waveform has {current_samples} samples, but target is {self.target_samples}. Truncating.")
            # Truncate as a fallback, though logic should prevent this.
            processed_wf = processed_wf[..., :self.target_samples]

        # Final check for shape
        if processed_wf.shape[1] != self.target_channels or processed_wf.shape[2] != self.target_samples:
            raise RuntimeError(f"Internal Preprocessing Error: Waveform shape {processed_wf.shape} "
                               f"does not match target ({current_batch_size}, {self.target_channels}, "
                               f"{self.target_samples}).")

        return processed_wf

    def get_aligned_waveforms(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """ Aligns both waveforms and returns them. """
        # Preprocess both waveform batches
        wf1_processed = self._align_one_batch(self.waveform1_orig, self.sr1)
        wf2_processed = self._align_one_batch(self.waveform2_orig, self.sr2)
        return wf1_processed, wf2_processed, self.target_sr
