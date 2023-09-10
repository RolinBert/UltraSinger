"""Plot transcribed data"""
import math
import os
from dataclasses import dataclass
from re import sub

import librosa
import numpy
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from modules.console_colors import ULTRASINGER_HEAD
from modules.Pitcher.pitched_data import PitchedData
from modules.Speech_Recognition.TranscribedData import TranscribedData


@dataclass
class PlottedNote:
    """Plotted note"""

    note: str
    frequency: float
    frequency_log_10: float
    octave: int


NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OCTAVES = [0, 1, 2, 3, 4, 5, 6]

def get_frequency_range(midi_note: str) -> float:
    midi = librosa.note_to_midi(midi_note)
    frequency_range = librosa.midi_to_hz(midi + 1) - librosa.midi_to_hz(midi)
    return frequency_range


def plot_notes(notes: list[str], octaves: list[int]) -> list[PlottedNote]:
    plotted_notes = []
    for octave in octaves:
        for note in notes:
            note_with_octave = note + str(octave)
            frequency = librosa.note_to_hz(note_with_octave)
            frequency_log_10 = numpy.log10([frequency])[0]
            plotted_notes.append(PlottedNote(note_with_octave, frequency, frequency_log_10, octave))
    
    return plotted_notes


PLOTTED_NOTES = plot_notes(NOTES, OCTAVES)

def plot(pitched_data: PitchedData,
         output_path: str,
         transcribed_data: list[TranscribedData] = None,
         midi_notes: list[str] = None,
         title: str = None,
         ) -> None:
    """Plot transcribed data"""

    # determine time between to datapoints if there is no gap (this is the step size crepe ran with)
    step_size = pitched_data.times[1]
    conf_t, conf_f, conf_c = get_confidence(pitched_data, 0.)

    if len(conf_f) < 2:
        print(f"{ULTRASINGER_HEAD} Plot can't be created; too few datapoints")
        return

    print(f"{ULTRASINGER_HEAD} Creating plot{': ' + title if title is not None else ''}")

    height = numpy.log10([max(conf_f)])[0] / 0.2
    width = pitched_data.times[-1] / 4

    plt.figure(1).set_figwidth(max(6.4, width))
    plt.figure(1).set_figheight(max(4, height))

    conf_f_log = numpy.log10(conf_f)
    conf_t_with_gaps, conf_f_log_with_gaps, conf_c_with_gaps = create_gaps(step_size, conf_t, conf_f_log, conf_c)

    # lower, upper = determine_bounds(conf_f_log)

    # if upper - lower >= 200:
    #     plt.ylim(max(0, (lower - 20)), upper + 20)
    # else:
    #     plt.ylim(0, max(upper + 20, 200))

    # plt.ylim(0, max(conf_f_log))
    ymin = max(0, min(conf_f_log) - 0.03)
    ymax = max(conf_f_log) + 0.03
    plt.ylim(ymin, ymax)
    plt.xlim(min(conf_t_with_gaps), max(conf_t_with_gaps))
    plt.plot(conf_t_with_gaps, conf_f_log_with_gaps, linewidth=0.1)

    scatter_path_collection = plt.scatter(conf_t_with_gaps, conf_f_log_with_gaps, s=5, c=conf_c_with_gaps, cmap=plt.cm.get_cmap('gray').reversed())
    plt.figure(1).colorbar(scatter_path_collection, label='confidence')

    if transcribed_data is not None:
        for i, data in enumerate(transcribed_data):
            note_frequency = librosa.note_to_hz(midi_notes[i])
            frequency_range = get_frequency_range(midi_notes[i])

            half_frequency_range = frequency_range / 2
            height = numpy.log10([note_frequency + half_frequency_range])[0] - numpy.log10([note_frequency - half_frequency_range])[0]

            xy_start_pos = (data.start, numpy.log10([note_frequency - half_frequency_range])[0])
            width = data.end - data.start
            rect = Rectangle(
                xy_start_pos,
                width,
                height,
                edgecolor='none',
                facecolor='red',
                alpha=0.5)
            plt.gca().add_patch(rect)

    plt.xlabel("Time (s)")
    plt.ylabel("log10 of Frequency (Hz)")
    dpi = 200

    notes_within_range = [x for x in PLOTTED_NOTES if ymin <= x.frequency_log_10 <= ymax]
    plt.yticks(
        [x.frequency_log_10 for x in notes_within_range],
        [x.note for x in notes_within_range],
    )
    plt.xticks(numpy.arange(math.floor(min(conf_t_with_gaps)), math.ceil(max(conf_t_with_gaps)), 5.0))

    for note in notes_within_range:
        color = 'b'
        if note.note.startswith('C') and not note.note.startswith('C#'):
            color = 'r'
        plt.axhline(y = note.frequency_log_10, color = color, linestyle = '-', linewidth = 0.2)

    if title is not None:
        plt.title(label=title)

    plt.figure(1).tight_layout(h_pad=1.4)
    plt.savefig(os.path.join(output_path, f"plot{'' if title is None else '_' + snake(title)}.svg"), dpi=dpi)
    plt.clf()
    plt.cla()


def get_confidence(
        pitched_data: PitchedData,
        threshold: float
) -> tuple[list[float], list[float], list[float]]:
    """Get high confidence data"""
    # todo: replace get_frequency_with_high_conf from pitcher
    conf_t = []
    conf_f = []
    conf_c = []
    for i in enumerate(pitched_data.times):
        pos = i[0]
        if pitched_data.confidence[pos] > threshold:
            conf_t.append(pitched_data.times[pos])
            conf_f.append(pitched_data.frequencies[pos])
            conf_c.append(pitched_data.confidence[pos])
    return conf_t, conf_f, conf_c

def create_gaps(
        step_size: float,
        conf_t: list[float],
        conf_f: list[float],
        conf_c: list[float]
) -> tuple[list[float], list[float], list[float]]:
    """Create gaps"""
    conf_t_with_gaps = []
    conf_f_with_gaps = []
    conf_c_with_gaps = []

    previous_time = 0
    for i, time in enumerate(conf_t):
        comes_right_before_next = False
        next_time = 0
        if i < (len(conf_t) - 1):
            next_time = conf_t[i+1]
            comes_right_before_next = next_time - time <= step_size

        comes_right_after_previous = time - previous_time <= step_size
        previous_frequency_is_not_gap = len(conf_f_with_gaps) > 0 and str(conf_f_with_gaps[-1]) != "nan"
        if previous_frequency_is_not_gap and time - previous_time > step_size:
            conf_t_with_gaps.append(time)
            conf_f_with_gaps.append(float("nan"))
            conf_c_with_gaps.append(conf_c[i])

        if (previous_frequency_is_not_gap and comes_right_after_previous) or comes_right_before_next:
            conf_t_with_gaps.append(time)
            conf_f_with_gaps.append(conf_f[i])
            conf_c_with_gaps.append(conf_c[i])

        previous_time = time

    return conf_t_with_gaps, conf_f_with_gaps, conf_c_with_gaps

def determine_bounds(
        conf_f_log: list[float]
) -> tuple[float, float]:
    """Determine bounds"""
    # calculate summary statistics
    data_mean = numpy.mean(conf_f_log)
    data_std = numpy.std(conf_f_log)
    # identify outliers
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off

    return lower, upper


def snake(s):
  return '_'.join(
    sub('([A-Z][a-z]+)', r' \1',
    sub('([A-Z]+)', r' \1',
    s.replace('-', ' '))).split()).lower()