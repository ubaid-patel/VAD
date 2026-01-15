# textgrid_utils.py
from praat_textgrids import TextGrid

def load_intervals(textgrid_path):
    """
    Returns list of (xmin, xmax, label)
    label: 1 = speech, 0 = silence/noise
    """
    tg = TextGrid(textgrid_path)
    tier = tg['silences']
    intervals = []

    for interval in tier:
        xmin = float(interval.xmin)
        xmax = float(interval.xmax)
        label = int(interval.text.strip())
        intervals.append((xmin, xmax, label))

    return intervals
