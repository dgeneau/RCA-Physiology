"""How each step-test type labels and bounds its movement-rate column.

A step test stores one rate per step in `rate_spm`, but what that number means
depends on the test: strokes per minute on an erg or on the water, pedal
revolutions per minute on a bike. The column id stays the same so the warehouse
schema does not fork; only the label, the unit and the plausible range change.

Kept out of pages/ so the entry form and the reporting editor read the same
table and cannot disagree on what a bike cadence of 95 is allowed to be.
"""

_ROWING = {
    "rate_label": "Stroke Rate",
    "rate_unit": "spm",
    "rate_range": (0, 80),
    # The split is derived from power with the Concept2 pace formula, which
    # only means something for rowing.
    "has_split": True,
}

STEP_TEST_PROFILES = {
    "erg_C2": _ROWING,
    "erg_RP3": _ROWING,
    "row": _ROWING,
    "bike": {
        "rate_label": "Cadence",
        "rate_unit": "rpm",
        "rate_range": (0, 200),
        "has_split": False,
    },
    "other": {
        "rate_label": "Rate",
        "rate_unit": "",
        "rate_range": (0, 200),
        "has_split": True,
    },
}

# Records with a missing or unrecognised test type get the loosest bounds, so
# nothing is flagged just because its type was never filled in.
DEFAULT_STEP_PROFILE = STEP_TEST_PROFILES["other"]


def step_profile(test_type):
    return STEP_TEST_PROFILES.get(test_type, DEFAULT_STEP_PROFILE)


def rate_header(test_type):
    """Column header, e.g. "Cadence (rpm)"."""
    p = step_profile(test_type)
    return f"{p['rate_label']} ({p['rate_unit']})" if p["rate_unit"] else p["rate_label"]


def rate_range_message(test_type):
    p = step_profile(test_type)
    low, high = p["rate_range"]
    unit = f" {p['rate_unit']}" if p["rate_unit"] else ""
    return f"{p['rate_label']} should be between {low:g} and {high:g}{unit}."


def rate_out_of_range(test_type, value):
    """True when a numeric rate falls outside the test type's bounds."""
    low, high = step_profile(test_type)["rate_range"]
    return value < low or value > high
