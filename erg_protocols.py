"""The erg protocols a result can belong to, shared by entry and reporting.

Distance used to be the identity of an erg test: a record was a 2k or a 6k and
nothing else, so `distance_m` alone said which piece it was and which other
results it could be compared against. A 30-minute test breaks that -- the clock
is fixed and the distance is the *result*, different for every athlete and every
attempt -- so the piece is named here instead, and distance is free to be
whatever was rowed.

Kept out of pages/entry.py so a plain script can import it without pulling in
Dash, and so the entry table and the reporting editor cannot drift apart on what
a protocol is called.
"""

PROTOCOL_2K = "2k"
PROTOCOL_6K = "6k"
PROTOCOL_30MIN = "30min"
PROTOCOL_PEAK_POWER = "peak_power"
PROTOCOL_CUSTOM = "custom"

# `distance_m` is prescribed for the two time trials and recorded for the timed
# piece; `time_min` is the other way round. The app fills in whichever one the
# protocol fixes, so a practitioner types only the result.
ERG_PROTOCOLS = (
    {"value": PROTOCOL_2K, "label": "2000 m", "distance_m": 2000, "time_min": None},
    {"value": PROTOCOL_6K, "label": "6000 m", "distance_m": 6000, "time_min": None},
    {"value": PROTOCOL_30MIN, "label": "30 min", "distance_m": None, "time_min": 30},
    {"value": PROTOCOL_PEAK_POWER, "label": "Peak power", "distance_m": None, "time_min": None},
    {"value": PROTOCOL_CUSTOM, "label": "Other", "distance_m": None, "time_min": None},
)

ERG_PROTOCOL_VALUES = tuple(p["value"] for p in ERG_PROTOCOLS)
ERG_PROTOCOL_LABELS = {p["value"]: p["label"] for p in ERG_PROTOCOLS}
ERG_PROTOCOL_OPTIONS = [{"label": p["label"], "value": p["value"]} for p in ERG_PROTOCOLS]

PROTOCOL_FIXED_DISTANCE = {
    p["value"]: p["distance_m"] for p in ERG_PROTOCOLS if p["distance_m"]
}
PROTOCOL_FIXED_TIME_MIN = {
    p["value"]: p["time_min"] for p in ERG_PROTOCOLS if p["time_min"]
}

# A peak power test is a handful of maximal strokes: the watts are the result
# and there is no meaningful distance or time to record. These protocols need
# only a power; distance and time are optional and stored as given.
POWER_ONLY_PROTOCOLS = frozenset({PROTOCOL_PEAK_POWER})

# What a spreadsheet column might call each one. Compared with case, spaces,
# underscores and punctuation stripped, the same way the upload headers are.
PROTOCOL_ALIASES = {
    PROTOCOL_2K: {"2k", "2km", "2000", "2000m", "2000merg", "2kerg", "2ktt"},
    PROTOCOL_6K: {"6k", "6km", "6000", "6000m", "6000merg", "6kerg", "6ktt"},
    PROTOCOL_30MIN: {
        "30min", "30mins", "30minute", "30minutes", "30minerg", "30minutepiece",
        "thirtymin", "thirtyminute", "30", "30mintest",
    },
    PROTOCOL_PEAK_POWER: {
        "peakpower", "peak", "peakw", "peakwatts", "maxpower", "pp", "peakpowertest",
    },
    PROTOCOL_CUSTOM: {"custom", "other", "misc"},
}

# A timed piece stopped a little early or run a little long is still a 30-minute
# test; a 6k that happens to take 23 minutes is not. The window is wide enough
# for the former and nowhere near the latter.
THIRTY_MIN_WINDOW_S = (28 * 60, 32 * 60)


def _simplify(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def normalize_protocol(value):
    """A protocol value out of whatever a spreadsheet or a person wrote."""
    if value in (None, ""):
        return None

    simplified = _simplify(value)
    if not simplified:
        return None

    for protocol, aliases in PROTOCOL_ALIASES.items():
        if simplified == protocol or simplified in aliases:
            return protocol

    return None


def infer_protocol(distance_m=None, total_seconds=None):
    """The protocol of a record that predates the field, or a CSV without it.

    Every erg record written before this field existed was a 2k or a 6k -- the
    schema allowed nothing else -- so distance decides those two outright. The
    duration check is for a fresh upload whose author left the column out.
    """
    try:
        distance = int(float(distance_m)) if distance_m not in (None, "") else None
    except (TypeError, ValueError):
        distance = None

    if distance == 2000:
        return PROTOCOL_2K
    if distance == 6000:
        return PROTOCOL_6K

    try:
        seconds = float(total_seconds) if total_seconds not in (None, "") else None
    except (TypeError, ValueError):
        seconds = None

    low, high = THIRTY_MIN_WINDOW_S
    if seconds is not None and low <= seconds <= high:
        return PROTOCOL_30MIN

    return PROTOCOL_CUSTOM


def protocol_label(value):
    return ERG_PROTOCOL_LABELS.get(value, value or "—")
