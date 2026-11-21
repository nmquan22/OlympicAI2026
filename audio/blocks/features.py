"""WAV -> mono/resampled waveform -> HTK-style log-mel. Requires scipy + torch."""
from math import gcd
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import torch


def read_wav(path, target_rate=16000):
    """Mono averaging discards spatial cues; keep channels for DCASE spatial tasks."""
    sample_rate, signal = wavfile.read(path)
    if signal.dtype == np.uint8:
        values = (signal.astype(np.float32) - 128) / 128
    elif np.issubdtype(signal.dtype, np.integer):
        values = signal.astype(np.float32) / (float(np.iinfo(signal.dtype).max) + 1)
    else:
        values = signal.astype(np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if len(values) == 0:
        raise ValueError("Empty audio file")
    divisor = gcd(sample_rate, target_rate)
    values = resample_poly(values, target_rate // divisor, sample_rate // divisor)
    return torch.from_numpy(np.asarray(values, dtype=np.float32).copy())


def log_mel(waveform, sample_rate=16000, n_fft=512, hop=160, bands=64):
    """Unbatched [samples] -> [mel,frames], power in decibels, constant padding."""
    mel_max = 2595 * np.log10(1 + sample_rate / 2 / 700)
    mel = torch.linspace(0, mel_max, bands + 2, device=waveform.device, dtype=waveform.dtype)
    edges = 700 * (10 ** (mel / 2595) - 1)
    frequencies = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, device=waveform.device, dtype=waveform.dtype)
    rising = (frequencies[None] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
    falling = (edges[2:, None] - frequencies[None]) / (edges[2:] - edges[1:-1])[:, None]
    filters = torch.minimum(rising, falling).clamp_min(0)
    window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
    power = torch.stft(waveform, n_fft, hop_length=hop, window=window,
                       return_complex=True, center=True, pad_mode="constant").abs().square()
    return 10 * (filters @ power).clamp_min(1e-10).log10()


def spec_augment(features, frequency_width=8, time_width=12):
    """Unbatched [mel,time]. Use only on training; zero means centered features."""
    result = features.clone()
    for axis, maximum in [(0, frequency_width), (1, time_width)]:
        width = int(torch.randint(min(maximum, result.shape[axis]) + 1, ()).item())
        start = int(torch.randint(result.shape[axis] - width + 1, ()).item())
        result.narrow(axis, start, width).zero_()
    return result
