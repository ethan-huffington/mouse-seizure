"""A dense 1-D residual U-Net: the architectural alternative to the feature pipeline.

The feature-based detector classifies a 2 s window at a time, then spends an
entire post-processing stage reconstructing the time resolution that windowing
threw away. This model does the opposite: it reads a stretch of raw signal and
emits one probability per *sample*, so there is no windowing, no overlap-add, and
no reconciliation step.

That is the whole point of the comparison. Both paths are scored through the
identical event-formation and evaluation harness, so the difference measured is
the architecture rather than the metric.

Kept deliberately small. With 8 animals, capacity is not the constraint --
generalising across animals is -- and a large model would be slower to train
without being more informative.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mouse_seizure.config import TARGET_FS


def best_device() -> torch.device:
    """MPS on Apple silicon, else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ResidualBlock(nn.Module):
    """Two convolutions with a skip connection around them."""

    def __init__(self, channels_in: int, channels_out: int, kernel: int = 9) -> None:
        super().__init__()
        pad = kernel // 2
        self.body = nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel, padding=pad),
            nn.BatchNorm1d(channels_out),
            nn.GELU(),
            nn.Conv1d(channels_out, channels_out, kernel, padding=pad),
            nn.BatchNorm1d(channels_out),
        )
        self.skip = (
            nn.Identity()
            if channels_in == channels_out
            else nn.Conv1d(channels_in, channels_out, 1)
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.skip(x))


class UNet1D(nn.Module):
    """Encoder-decoder with skip connections, one output per input sample.

    The encoder halves resolution four times, so the deepest layer sees a
    receptive field of several seconds -- wide enough to judge whether a rhythm
    has the sustained regularity of a seizure rather than a passing oscillation.
    Skip connections carry the fine detail back so onsets stay sharp.
    """

    def __init__(self, widths: tuple[int, ...] = (16, 32, 64, 128)) -> None:
        super().__init__()
        self.depth = len(widths)
        self.downs = nn.ModuleList()
        channels = 1
        for width in widths:
            self.downs.append(ResidualBlock(channels, width))
            channels = width
        self.pool = nn.MaxPool1d(2)
        self.bottom = ResidualBlock(channels, channels * 2)

        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        channels *= 2
        for width in reversed(widths):
            self.ups.append(nn.ConvTranspose1d(channels, width, 2, stride=2))
            self.up_blocks.append(ResidualBlock(width * 2, width))
            channels = width
        self.head = nn.Conv1d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.downs:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottom(x)
        for up, block, skip in zip(self.ups, self.up_blocks, reversed(skips)):
            x = up(x)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x).squeeze(1)


def dice_bce_loss(
    logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 0.5
) -> torch.Tensor:
    """Dice plus binary cross-entropy.

    Cross-entropy alone is dominated by the ~96% negative samples and drifts
    toward predicting nothing. Dice measures overlap between predicted and true
    regions, so it stays sensitive to a rare positive class; keeping both means
    the model is pushed on region overlap without losing per-sample calibration.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    dice = 1.0 - ((2 * intersection + 1.0) / (union + 1.0)).mean()
    return (1 - dice_weight) * bce + dice_weight * dice


class SegmentSampler:
    """Draws fixed-length segments from a set of memory-mapped recordings.

    Half the segments are drawn so they contain an annotated seizure. Sampling
    uniformly instead would spend most of the budget on baseline, since seizures
    occupy about 4% of the signal; oversampling them balances gradient signal
    without changing any recording's contents.
    """

    def __init__(
        self,
        recordings: list[tuple[np.ndarray, np.ndarray]],
        segment_s: float = 20.0,
        fs: float = TARGET_FS,
        positive_fraction: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.recordings = recordings
        self.length = int(segment_s * fs)
        self.positive_fraction = positive_fraction
        self.rng = np.random.default_rng(seed)
        # Onset index of every annotated event, per recording.
        self.onsets = []
        for _, labels in recordings:
            y = np.asarray(labels)
            starts = np.flatnonzero(np.diff(np.concatenate(([0], y))) == 1)
            self.onsets.append(starts)

    def batch(self, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for _ in range(size):
            i = self.rng.integers(len(self.recordings))
            signal, labels = self.recordings[i]
            limit = len(signal) - self.length
            want_positive = self.rng.random() < self.positive_fraction
            onsets = self.onsets[i]

            if want_positive and len(onsets):
                onset = onsets[self.rng.integers(len(onsets))]
                # Place the event somewhere inside the segment, not always at its start.
                offset = self.rng.integers(0, self.length)
                start = int(np.clip(onset - offset, 0, limit))
            else:
                start = int(self.rng.integers(0, max(limit, 1)))

            xs.append(np.asarray(signal[start : start + self.length], dtype=np.float32))
            ys.append(np.asarray(labels[start : start + self.length], dtype=np.float32))

        return (
            torch.from_numpy(np.stack(xs)).unsqueeze(1),
            torch.from_numpy(np.stack(ys)),
        )


@torch.no_grad()
def predict_trace(
    model: UNet1D,
    signal: np.ndarray,
    device: torch.device,
    chunk_s: float = 20.0,
    fs: float = TARGET_FS,
    batch: int = 32,
) -> np.ndarray:
    """Per-sample probability across a whole recording, in chunks.

    Chunks are a multiple of 16 samples so the four pooling stages divide evenly;
    the tail is padded and trimmed rather than handled as a special case.
    """
    model.eval()
    length = int(chunk_s * fs) // 16 * 16
    n = len(signal)
    padded = int(np.ceil(n / length) * length)
    flat = np.zeros(padded, dtype=np.float32)
    flat[:n] = signal
    chunks = flat.reshape(-1, length)

    out = np.empty_like(chunks)
    for i in range(0, len(chunks), batch):
        block = torch.from_numpy(chunks[i : i + batch]).unsqueeze(1).to(device)
        out[i : i + batch] = torch.sigmoid(model(block)).cpu().numpy()
    return out.reshape(-1)[:n]
