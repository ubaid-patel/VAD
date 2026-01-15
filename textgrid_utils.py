# textgrid_utils.py
def load_intervals(textgrid_path):
    """
    Parses Praat TextGrid (IntervalTier)
    Returns list of (xmin, xmax, label)
    label: 1 = speech, 0 = silence/noise
    """
    intervals = []

    with open(textgrid_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    xmin = xmax = None

    for line in lines:
        line = line.strip()

        if line.startswith("xmin ="):
            xmin = float(line.split("=")[1])

        elif line.startswith("xmax ="):
            xmax = float(line.split("=")[1])

        elif line.startswith('text ='):
            label = int(line.split('"')[1])

            if xmin is not None and xmax is not None:
                intervals.append((xmin, xmax, label))
                xmin = xmax = None

    return intervals
