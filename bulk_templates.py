"""Column layouts and worked examples for the two bulk-upload CSV templates.

Kept out of pages/entry.py so a plain script can import it without pulling in
Dash, the page registry and a warehouse client — scripts/generate_templates.py
writes templates/*.csv from exactly the definitions the app's download buttons
serve, so the file a practitioner was emailed and the file the app hands out
can never drift apart.
"""

import csv
import io

# One row per step. Session-level columns (athlete .. notes) repeat on every
# step row of that test: fill-down in a spreadsheet is a single drag, and a
# stray blank line cannot silently re-attribute steps to the athlete above.
STEP_TEMPLATE_COLUMNS = (
    "athlete",
    "profile_id",
    "test_date",
    "body_mass_kg",
    "test_type",
    "mode",
    "notes",
    "step_no",
    "step_type",
    "target_power_w",
    "actual_power_w",
    "heart_rate_bpm",
    "lactate_mmol",
    "vo2",
    "stroke_rate_spm",
    "rpe",
    "time_s",
)

# Two athletes, so the repeated session block is visible rather than implied,
# and a Max final step, so the Submax/Max column has a reason to exist. The
# placeholder names show the expected "First Last" shape, and the second one
# carries a hyphen to show that punctuation in a real surname is fine.
STEP_TEMPLATE_EXAMPLE_ROWS = (
    {
        "athlete": "Example Athlete", "test_date": "2026-09-15",
        "body_mass_kg": 72.4, "test_type": "erg_C2", "mode": "Submax",
        "notes": "Pre-camp step test", "step_no": 1, "step_type": "Submax",
        "target_power_w": 150, "actual_power_w": 152, "heart_rate_bpm": 128,
        "lactate_mmol": 1.2, "vo2": "", "stroke_rate_spm": 20, "rpe": 11,
        "time_s": 240,
    },
    {
        "athlete": "Example Athlete", "test_date": "2026-09-15",
        "body_mass_kg": 72.4, "test_type": "erg_C2", "mode": "Submax",
        "notes": "Pre-camp step test", "step_no": 2, "step_type": "Submax",
        "target_power_w": 180, "actual_power_w": 181, "heart_rate_bpm": 141,
        "lactate_mmol": 1.8, "vo2": "", "stroke_rate_spm": 22, "rpe": 13,
        "time_s": 240,
    },
    {
        "athlete": "Example Athlete", "test_date": "2026-09-15",
        "body_mass_kg": 72.4, "test_type": "erg_C2", "mode": "Submax",
        "notes": "Pre-camp step test", "step_no": 3, "step_type": "Max",
        "target_power_w": 240, "actual_power_w": 244, "heart_rate_bpm": 178,
        "lactate_mmol": 6.4, "vo2": "", "stroke_rate_spm": 30, "rpe": 19,
        "time_s": 240,
    },
    {
        "athlete": "Sample Athlete-Two", "test_date": "2026-09-15",
        "body_mass_kg": 84.1, "test_type": "erg_C2", "mode": "Submax",
        "notes": "", "step_no": 1, "step_type": "Submax",
        "target_power_w": 200, "actual_power_w": 199, "heart_rate_bpm": 122,
        "lactate_mmol": 1.1, "vo2": "", "stroke_rate_spm": 19, "rpe": 10,
        "time_s": 240,
    },
    {
        "athlete": "Sample Athlete-Two", "test_date": "2026-09-15",
        "body_mass_kg": 84.1, "test_type": "erg_C2", "mode": "Submax",
        "notes": "", "step_no": 2, "step_type": "Submax",
        "target_power_w": 240, "actual_power_w": 242, "heart_rate_bpm": 139,
        "lactate_mmol": 1.9, "vo2": "", "stroke_rate_spm": 21, "rpe": 13,
        "time_s": 240,
    },
)

# One row per athlete per piece.
ERG_TEMPLATE_COLUMNS = (
    "athlete",
    "profile_id",
    "test_date",
    "protocol",
    "distance_m",
    "stroke_rate_spm",
    "power_w",
    "time_min",
    "time_s",
)

# time_min/time_s are whole minutes plus the leftover seconds, i.e. 7:12.4 is
# time_min 7, time_s 12.4 — not 7 and 432.4.
#
# The third and fourth rows are 30-minute pieces: the clock is the constant and
# the distance is the result, which is why distance_m differs between two
# athletes who rowed the same test. Leaving protocol blank is allowed — it is
# read back from the distance and duration — but a 30-minute piece is worth
# naming, since only the protocol tells it apart from an unusual time trial.
ERG_TEMPLATE_EXAMPLE_ROWS = (
    {
        "athlete": "Example Athlete", "test_date": "2026-09-15",
        "protocol": "2k", "distance_m": 2000, "stroke_rate_spm": 32,
        "power_w": 245, "time_min": 7, "time_s": 12.4,
    },
    {
        "athlete": "Example Athlete", "test_date": "2026-09-15",
        "protocol": "6k", "distance_m": 6000, "stroke_rate_spm": 26,
        "power_w": 205, "time_min": 23, "time_s": 4.8,
    },
    {
        "athlete": "Example Athlete", "test_date": "2026-09-22",
        "protocol": "30min", "distance_m": 8420, "stroke_rate_spm": 24,
        "power_w": 212, "time_min": 30, "time_s": 0,
    },
    {
        "athlete": "Sample Athlete-Two", "test_date": "2026-09-22",
        "protocol": "30min", "distance_m": 9015, "stroke_rate_spm": 25,
        "power_w": 258, "time_min": 30, "time_s": 0,
    },
    {
        "athlete": "Sample Athlete-Two", "test_date": "2026-09-15",
        "protocol": "2k", "distance_m": 2000, "stroke_rate_spm": 34,
        "power_w": 398, "time_min": 6, "time_s": 8.1,
    },
    {
        "athlete": "Sample Athlete-Two", "test_date": "2026-09-15",
        "protocol": "peak_power", "distance_m": "", "stroke_rate_spm": "",
        "power_w": 812, "time_min": "", "time_s": "",
    },
)


def template_csv(columns, rows):
    """Render a template as CSV text, blank for anything an example omits."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue()


def step_template_csv():
    return template_csv(STEP_TEMPLATE_COLUMNS, STEP_TEMPLATE_EXAMPLE_ROWS)


def erg_template_csv():
    return template_csv(ERG_TEMPLATE_COLUMNS, ERG_TEMPLATE_EXAMPLE_ROWS)
