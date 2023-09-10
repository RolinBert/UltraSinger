"""Pitcher module"""

import sys
import os

import torch
import torchcrepe
from torch.cuda import OutOfMemoryError

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted
from modules.Pitcher.pitched_data import PitchedData
from modules.plot import plot

DEFAULT_MODEL: str = "full"
SUPPORTED_MODELS: list[str] = ["tiny", "full"]
TIMES_DECIMAL_PLACES: int = 3


def get_pitch_with_crepe_file(
    filename: str,
    model: str,
    device: str = "cpu",
    batch_size: int = None,
    step_size: int = 10,
    frequency_boundary_lower: int = 0,
    frequency_boundary_upper: int = 2006,
) -> PitchedData:
    """Pitch with crepe"""

    # Load audio
    audio, sample_rate = torchcrepe.load.audio(filename)

    output_path = os.path.abspath(os.path.dirname(os.path.abspath(filename)))

    return get_pitch_with_crepe(
        audio, sample_rate, model, device, batch_size, step_size, output_path=output_path, frequency_boundary_lower=frequency_boundary_lower, frequency_boundary_upper=frequency_boundary_upper
    )


def get_pitch_with_crepe(
    audio,
    sample_rate: int,
    model: str,
    device: str = "cpu",
    batch_size: int = None,
    step_size: int = 10,
    frequency_boundary_lower: int = 0,
    frequency_boundary_upper: int = 2006,
    output_path: str = ".",
) -> PitchedData:
    """Pitch with crepe"""

    if model not in SUPPORTED_MODELS:
        print(
            f"{ULTRASINGER_HEAD} {blue_highlighted('crepe')} model {blue_highlighted(model)} is not supported, defaulting to {blue_highlighted(DEFAULT_MODEL)}"
        )
        model = DEFAULT_MODEL

    print(
        f"{ULTRASINGER_HEAD} Pitching using {blue_highlighted('crepe')} with model {blue_highlighted(model)} and {red_highlighted(device)} as worker"
    )

    step_size_seconds = round(step_size / 1000, TIMES_DECIMAL_PLACES)
    steps_per_second = 1 / step_size_seconds
    hop_length = sample_rate // steps_per_second

    frequencies_tensor = []
    confidence_tensor = []
    try:
        frequencies_tensor, confidence_tensor = torchcrepe.predict(
            audio,
            sample_rate,
            hop_length,
            frequency_boundary_lower,
            frequency_boundary_upper,
            model,
            return_periodicity=True,
            batch_size=batch_size,
            device=device,
        )
    except OutOfMemoryError as oom_exception:
        print(oom_exception)
        print(
            f"{ULTRASINGER_HEAD} {blue_highlighted('crepe')} ran out of GPU memory; reduce --crepe_batch_size or force cpu with --force_crepe_cpu"
        )
        sys.exit(1)


    plot(tensors_to_pitched_data(frequencies_tensor, confidence_tensor, step_size_seconds), output_path, title="0 prediction")

    confidence_tensor = torchcrepe.threshold.Silence(-60.)(confidence_tensor,
                                                 audio,
                                                 sample_rate,
                                                 hop_length)
    plot(tensors_to_pitched_data(frequencies_tensor, confidence_tensor, step_size_seconds), output_path, title="1 silence removal")

    # We'll use a 30 millisecond window assuming a hop length of 10 milliseconds
    win_length = 3

    # Median filter noisy confidence value
    confidence_tensor = torchcrepe.filter.median(confidence_tensor, win_length)
    plot(tensors_to_pitched_data(frequencies_tensor, confidence_tensor, step_size_seconds), output_path, title="2 noisy confidence removal")

    # Remove inharmonic regions
    # frequencies_tensor = torchcrepe.threshold.At(.21)(frequencies_tensor, confidence_tensor)
    # plot(tensors_to_pitched_data(frequencies_tensor, confidence_tensor, step_size_seconds), output_path, title="3 inharmonic region removal")

    # Optionally smooth pitch to remove quantization artifacts
    frequencies_tensor = torchcrepe.filter.mean(frequencies_tensor, win_length)
    plot(tensors_to_pitched_data(frequencies_tensor, confidence_tensor, step_size_seconds), output_path, title="4 quantization artifacts removal")

    return tensors_to_pitched_data(frequencies_tensor, confidence_tensor, step_size_seconds)


def tensors_to_pitched_data(frequencies_tensor: torch.Tensor, confidence_tensor: torch.Tensor, step_size_seconds: float) -> PitchedData:
    """Convert tensors to plain arrays"""
    frequencies = frequencies_tensor.detach().cpu().numpy().squeeze(0)
    confidence = confidence_tensor.detach().cpu().numpy().squeeze(0)
    times = [
        round(i * step_size_seconds, TIMES_DECIMAL_PLACES)
        for i, x in enumerate(confidence)
    ]
    return PitchedData(times, frequencies, confidence)


def get_frequency_with_high_confidence(
    frequencies: list[float], confidences: list[float], threshold=0.
) -> list[float]:
    """Get frequency with high confidence"""
    conf_f = []
    for i, conf in enumerate(confidences):
        if conf > threshold:
            conf_f.append(frequencies[i])
    if not conf_f:
        conf_f = frequencies
    return conf_f


class Pitcher:
    """Docstring"""
