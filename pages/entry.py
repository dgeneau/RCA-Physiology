
# pages/entry.py

from datetime import datetime, date
import pandas as pd
import numpy as np
import scipy as sp

import dash
from dash import html, dcc, dash_table, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash_auth_external.exceptions import TokenExpiredError

import plotly.express as px
import plotly.graph_objects as go

from auth_setup import auth
from utils import fetch_profiles, decode_csv_bytes

from settings import SITE_URL, VO2_STEP_SOURCE_UUID, ERG_TEST_SOURCE_UUID
from erg_protocols import (
    ERG_PROTOCOL_OPTIONS,
    ERG_PROTOCOL_VALUES,
    PROTOCOL_FIXED_DISTANCE,
    PROTOCOL_FIXED_TIME_MIN,
    PROTOCOL_PEAK_POWER,
    POWER_ONLY_PROTOCOLS,
    infer_protocol,
    normalize_protocol,
    protocol_label,
)
from bulk_templates import (
    ERG_TEMPLATE_COLUMNS,
    STEP_TEMPLATE_COLUMNS,
    erg_template_csv,
    step_template_csv,
)
from warehouse import WarehouseAPIConfig, WarehouseClient, WarehouseClientError
from lactate_thresholds import LA_2, LA_4, compute_hr_zones, fit_lactate_curve, power_at_lactate
from step_protocols import (
    is_submax_test,
    manual_max_issues,
    takes_manual_max_results,
    rate_header,
    rate_out_of_range,
    rate_range_message,
    step_profile,
)
import base64
import hashlib
import io
import json
import re

cfg = WarehouseAPIConfig(base_url=SITE_URL)
wc = WarehouseClient(cfg, token_getter=auth.get_token)

dash.register_page(__name__, path="/entry", name="Data Entry")

# Every athlete list on this page is drawn from one org.
SPORT_ORG_ID = 13

AUTH_KEEPALIVE_INTERVAL_MS = 4 * 60 * 1000
MISSING_ERG_NUMERIC_VALUE = 0.1

# Entered data survives a refresh, a mis-click on the navbar, and the
# token-expiry redirect. "session" (not "local") is deliberate: a fresh tab
# starts clean, so a shared erg-room tablet never shows the previous
# practitioner's athlete and risk mis-attributing a test.
PERSISTENCE_TYPE = "session"


# =========================================================
# STEP TEST TABLE
# =========================================================
DEFAULT_TEST_TYPE = "erg_C2"
DEFAULT_MODE = "Submax"
DEFAULT_STEP_ROW_COUNT = 3


def blank_step_row(step_no=None, mode=DEFAULT_MODE):
    return {
        "step_no": step_no,
        "Type": mode,
        "T_PO": None,
        "A_PO": None,
        "HR": None,
        "La": None,
        "V02": None,
        "rate": None,
        "split": None,
        "rpe": None,
        "time_s": None,
    }


def blank_step_rows(n=DEFAULT_STEP_ROW_COUNT, mode=DEFAULT_MODE):
    """Fresh row dicts. A factory, not a constant, so Reset and the layout
    never hand callbacks the same mutable objects."""
    return [blank_step_row(i + 1, mode) for i in range(n)]

TABLE_COLUMNS = [
    {"name": "Step Number", "id": "step_no", "type": "numeric"},
    {"name": "Submax/Max", "id": "Type", "type": "text"},
    {"name": "Target power (W)", "id": "T_PO", "type": "numeric"},
    {"name": "Power (W)", "id": "A_PO", "type": "numeric"},
    {"name": "HR (bpm)", "id": "HR", "type": "numeric"},
    {"name": "Lactate (mmol/L)", "id": "La", "type": "numeric"},
    {"name": "VO₂ (L/min)", "id": "V02", "type": "numeric"},
    {"name": "Rate", "id": "rate", "type": "numeric"},
    {"name": "Split time", "id": "split", "type": "numeric", "editable": False},
    {"name": "RPE", "id": "rpe", "type": "numeric"},
    {"name": "Time in Step (s)", "id": "time_s", "type": "numeric"},
]


# =========================================================
# ERG TEST TABLE (each row = one athlete)
# =========================================================
ERG_DEFAULT_ROW_COUNT = 3


def blank_erg_row(row_no=None, test_date=None):
    return {
        "row_no": row_no,
        "profile_id": "",
        "test_date": test_date or date.today().isoformat(),
        "protocol": None,
        "distance_m": None,
        "stroke_rate_spm": None,
        "power_w": None,
        "time_min": None,
        "time_s": None,
    }


def blank_erg_rows(n=ERG_DEFAULT_ROW_COUNT):
    return [blank_erg_row(i + 1) for i in range(n)]

ERG_TABLE_COLUMNS = [
    {"name": "Row", "id": "row_no", "type": "numeric"},
    {"name": "Test Date", "id": "test_date", "type": "text"},
    {"name": "Test", "id": "protocol", "type": "text", "presentation": "dropdown"},
    {"name": "Athlete", "id": "profile_id", "type": "text", "presentation": "dropdown"},
    {"name": "Distance (m)", "id": "distance_m", "type": "numeric"},
    {"name": "Stroke Rate (spm)", "id": "stroke_rate_spm", "type": "numeric"},
    {"name": "Power (W)", "id": "power_w", "type": "numeric"},
    {"name": "Time (min)", "id": "time_min", "type": "numeric"},
    {"name": "Time (s)", "id": "time_s", "type": "numeric"},
]

ERG_UPLOAD_COLUMN_ALIASES = {
    "row_no": {"rowno", "row", "rownumber"},
    "profile_id": {"profileid", "athleteid", "subjectid"},
    "athlete": {"athlete", "name", "fullname", "athletename", "about"},
    "test_date": {"testdate", "date", "testday"},
    "protocol": {"protocol", "test", "testname", "piece", "event", "ergtest"},
    "distance_m": {"distancem", "distance", "metres", "meters", "ergdistance"},
    "stroke_rate_spm": {
        "strokeratespm",
        "strokerate",
        "rate",
        "averagestrokerate",
        "avgrate",
        "spm",
    },
    "power_w": {"powerw", "power", "watts", "avgpower", "averagepower"},
    "time_min": {"timemin", "timeminutes", "minutes", "elapsedmin", "elapsedminutes"},
    "time_s": {"times", "timesec", "timeseconds", "seconds", "elapsedsec", "elapsedseconds"},
    "time": {"time", "elapsedtime", "resulttime", "score"},
}

# =========================================================
# BULK STEP TEST UPLOAD FORMAT
# =========================================================
# Headers are matched with case, spaces, underscores and punctuation ignored,
# so "Heart Rate", "heart_rate_bpm" and "HR" all land on the same field. That
# is what lets a practitioner upload the sheet they already keep instead of
# re-typing it into ours.
STEP_UPLOAD_COLUMN_ALIASES = {
    "profile_id": {"profileid", "athleteid", "subjectid"},
    "athlete": {"athlete", "athletename", "name", "fullname", "about", "rower"},
    "test_date": {"testdate", "date", "testday", "sessiondate"},
    "body_mass_kg": {
        "bodymasskg", "bodymass", "bodyweightkg", "bodyweight",
        "weightkg", "weight", "masskg", "mass",
    },
    "test_type": {"testtype", "modality", "ergtype", "equipment"},
    "mode": {"mode", "submaxmax", "maxsubmax", "testmode"},
    "notes": {"notes", "note", "comments", "comment"},
    "step_no": {"stepno", "step", "stepnumber", "stage", "stageno", "stagenumber"},
    "step_type": {"steptype", "stagetype", "rowtype"},
    "target_power_w": {
        "targetpowerw", "targetpower", "targetpo", "targetwatts", "tpo",
    },
    "actual_power_w": {
        "actualpowerw", "actualpower", "actualpo", "power", "powerw",
        "watts", "po", "apo",
    },
    "heart_rate_bpm": {"heartratebpm", "heartrate", "hr", "hrbpm", "bpm"},
    "lactate_mmol": {"lactatemmol", "lactate", "bloodlactate", "la", "bla"},
    "vo2": {"vo2", "v02", "vo2lmin", "vo2absolute"},
    "stroke_rate_spm": {"strokeratespm", "strokerate", "rate", "spm", "cadence"},
    "rpe": {"rpe", "perceivedexertion", "borg"},
    "time_s": {
        "times", "timeins", "timeinstep", "timeinsteps", "stepduration",
        "stepdurations", "duration", "durations", "timeseconds",
    },
}

# An athlete is named by either column, so neither is listed here; the parser
# reports a row that carries neither.
STEP_UPLOAD_REQUIRED_COLUMNS = ("test_date", "step_no")

STEP_NUMERIC_FIELDS = {
    "body_mass_kg": "body mass",
    "target_power_w": "target power",
    "actual_power_w": "actual power",
    "heart_rate_bpm": "heart rate",
    "lactate_mmol": "lactate",
    "vo2": "VO2",
    "stroke_rate_spm": "stroke rate",
    "time_s": "time in step",
}

STEP_MODE_VALUES = {"max": "Max", "submax": "Submax"}

STEP_TEST_TYPE_VALUES = ("erg_C2", "erg_RP3", "row", "bike", "other")

BULK_STEP_PREVIEW_COLUMNS = [
    {"name": "Athlete", "id": "athlete", "type": "text"},
    {"name": "Profile ID", "id": "profile_id", "type": "numeric"},
    {"name": "Test Date", "id": "test_date", "type": "text"},
    {"name": "Test Type", "id": "test_type", "type": "text"},
    {"name": "Mode", "id": "mode", "type": "text"},
    {"name": "Steps", "id": "steps", "type": "numeric"},
    {"name": "Body Mass (kg)", "id": "body_mass_kg", "type": "numeric"},
    {"name": "Session ID", "id": "session_id", "type": "text"},
]


# =========================================================
# ZONES TABLE
# =========================================================
ZONES_DEFAULT_ROWS = [
    {"Zone": "Z1", "HR_low": None, "HR_high": None, "PO_low": None, "PO_high": None,
     "Split_low": None, "Split_high": None, "Rate_low": None, "Rate_high": None, "Notes": ""},
    {"Zone": "Z2", "HR_low": None, "HR_high": None, "PO_low": None, "PO_high": None,
     "Split_low": None, "Split_high": None, "Rate_low": None, "Rate_high": None, "Notes": ""},
    {"Zone": "Z3", "HR_low": None, "HR_high": None, "PO_low": None, "PO_high": None,
     "Split_low": None, "Split_high": None, "Rate_low": None, "Rate_high": None, "Notes": ""},
    {"Zone": "Z4", "HR_low": None, "HR_high": None, "PO_low": None, "PO_high": None,
     "Split_low": None, "Split_high": None, "Rate_low": None, "Rate_high": None, "Notes": ""},
    {"Zone": "Z5", "HR_low": None, "HR_high": None, "PO_low": None, "PO_high": None,
     "Split_low": None, "Split_high": None, "Rate_low": None, "Rate_high": None, "Notes": ""},
]

ZONES_COLUMNS = [
    {"name": "Zone", "id": "Zone", "type": "text"},
    {"name": "HR Low", "id": "HR_low", "type": "numeric"},
    {"name": "HR High", "id": "HR_high", "type": "numeric"},
    {"name": "PO Low (W)", "id": "PO_low", "type": "numeric"},
    {"name": "PO High (W)", "id": "PO_high", "type": "numeric"},
    {"name": "Split Low (s/500)", "id": "Split_low", "type": "text"},
    {"name": "Split High (s/500)", "id": "Split_high", "type": "text"},
    {"name": "Rate Low (spm)", "id": "Rate_low", "type": "numeric"},
    {"name": "Rate High (spm)", "id": "Rate_high", "type": "numeric"},
    {"name": "Notes", "id": "Notes", "type": "text"},
]


def step_table_columns(test_type):
    """TABLE_COLUMNS as the chosen test type shows them: the rate column is
    "Cadence (rpm)" on a bike, and the rowing split is dropped where it means
    nothing. Ids never change, so the stored record shape is the same."""
    profile = step_profile(test_type)
    columns = []
    for col in TABLE_COLUMNS:
        if col["id"] == "split" and not profile["has_split"]:
            continue
        col = dict(col)
        if col["id"] == "rate":
            col["name"] = rate_header(test_type)
        columns.append(col)
    return columns


def zones_table_columns(test_type):
    profile = step_profile(test_type)
    unit = f" ({profile['rate_unit']})" if profile["rate_unit"] else ""
    columns = []
    for col in ZONES_COLUMNS:
        if col["id"] in ("Split_low", "Split_high") and not profile["has_split"]:
            continue
        col = dict(col)
        if col["id"] == "Rate_low":
            col["name"] = f"{profile['rate_label']} Low{unit}"
        elif col["id"] == "Rate_high":
            col["name"] = f"{profile['rate_label']} High{unit}"
        columns.append(col)
    return columns


def rate_range_styles(test_type):
    """Flag a rate cell red as soon as it leaves the test type's range."""
    low, high = step_profile(test_type)["rate_range"]
    return [
        {
            "if": {"filter_query": f"{{rate}} < {low} || {{rate}} > {high}", "column_id": "rate"},
            "backgroundColor": "#f8d7da",
            "border": "1px solid #dc3545",
        }
    ]


# =========================================================
# HELPERS
# =========================================================
def make_card(title, body):
    return dbc.Card(
        [dbc.CardHeader(html.B(title)), dbc.CardBody(body)],
        className="shadow-sm",
    )


def auth_relogin_message(action="continue"):
    return (
        f"Your login session has expired. Re-authenticate in another tab, then return here to {action}. "
        "The data currently entered in this page should remain in the table while this tab stays open."
    )


def is_auth_error(exc):
    if isinstance(exc, TokenExpiredError):
        return True
    if isinstance(exc, ValueError) and "token" in str(exc).lower():
        return True
    if isinstance(exc, WarehouseClientError) and "token" in str(exc).lower():
        return True
    return False


def to_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def simplify_header(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def normalize_person_name(value):
    """Lower-cased, whitespace-collapsed name — 'JOHN   DOE' -> 'john doe'."""
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def loose_person_name(value):
    """Letters and digits only, so hyphens, apostrophes and spacing stop mattering."""
    return re.sub(r"[^a-z0-9]+", "", normalize_person_name(value))


# Fields that define a submission's identity. session_id/session_ts are
# excluded on purpose: they are regenerated on every click, so including them
# would make every duplicate look unique.
FINGERPRINT_FIELDS = (
    "profile_id",
    "test_date",
    "body_mass_kg",
    "test_type",
    "mode",
    "notes",
    "step_no",
    "step_type",
    "target_po_w",
    "actual_po_w",
    "hr_bpm",
    "lactate_mmol",
    "vo2",
    "rate_spm",
    "split_sec_per_500",
    "rpe",
    "time_s",
    "map_w",
    "vo2max",
)


def draft_is_restorable(rows):
    """True for any real saved table state.

    Deliberately not a "did they type anything" test: the draft also carries
    the row *count*, so a practitioner who deleted down to one row or added a
    ninth step gets that back rather than the three default rows. Restoring a
    draft of blank rows over blank rows is a harmless no-op; losing their row
    structure is not.
    """
    return (
        isinstance(rows, list)
        and len(rows) > 0
        and all(isinstance(row, dict) for row in rows)
    )


def submission_fingerprint(records):
    """Stable hash of the data being ingested, ignoring per-click timestamps.

    Lets us recognise a second click that would push the exact same test again
    under a brand new session_id, which nothing downstream could dedupe.
    """
    trimmed = [
        {k: to_float(r.get(k)) if k not in ("test_type", "mode", "notes", "step_type", "test_date") else r.get(k)
         for k in FINGERPRINT_FIELDS}
        for r in records
    ]
    blob = json.dumps(trimmed, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------
# SHARED CSV VALUE PARSING
# --------------------------------------------------

def clean_value(x):
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x = x.strip()
        return x if x != "" else None
    return x


def parse_rpe(x):
    """Parse RPE fields like '2- Fairly light' -> 2."""
    x = clean_value(x)
    if x is None:
        return None

    if isinstance(x, (int, float)) and not pd.isna(x):
        return float(x)

    s = str(x).strip()
    if "-" in s:
        s = s.split("-", 1)[0].strip()

    try:
        return float(s)
    except Exception:
        return None


def map_test_type(test_type, other_test_type=None):
    test_type = clean_value(test_type)
    other_test_type = clean_value(other_test_type)

    mapping = {
        "Erg C2": "erg_C2",
        "Erg RP3": "erg_RP3",
        "On-Water": "row",
        "Bike": "bike",
        "Other": "other",
    }

    if test_type in mapping:
        return mapping[test_type]

    if test_type:
        tt = str(test_type).lower()
        if "c2" in tt:
            return "erg_C2"
        if "rp3" in tt:
            return "erg_RP3"
        if "water" in tt or "row" in tt:
            return "row"
        if "bike" in tt:
            return "bike"
        if "other" in tt:
            return "other"

    if other_test_type:
        return "other"

    return None


def canonical_columns(columns, alias_map):
    """Map a sheet's headers onto our field names.

    A field is claimed by the first header that matches it, so a sheet
    carrying both "Power" and "Actual PO" keeps the leftmost rather than
    letting the rename silently collapse the two into one column.
    """
    canonical = {}
    used = set()
    for column in columns:
        simplified = simplify_header(column)
        for target, aliases in alias_map.items():
            if target not in used and simplified in aliases:
                canonical[column] = target
                used.add(target)
                break
    return canonical


def canonical_erg_upload_columns(columns):
    return canonical_columns(columns, ERG_UPLOAD_COLUMN_ALIASES)


def parse_time_to_seconds(value):
    value = clean_value(value)
    if value is None:
        return None

    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)
        return float(text)
    except ValueError:
        return None


def split_total_seconds(total_seconds):
    total_seconds = to_float(total_seconds)
    if total_seconds is None:
        return None, None

    minutes = int(total_seconds // 60)
    seconds = round(total_seconds - (minutes * 60), 2)
    if seconds >= 60:
        minutes += 1
        seconds = round(seconds - 60, 2)
    return minutes, seconds


def normalize_erg_time_parts(time_min=None, time_s=None, time_value=None, split_sec_per_500=None, distance=None):
    minute_part = pd.to_numeric(time_min, errors="coerce")
    second_part = pd.to_numeric(time_s, errors="coerce")

    if pd.notna(minute_part) and pd.notna(second_part):
        return int(float(minute_part)), round(float(second_part), 2)

    total_seconds = parse_time_to_seconds(time_value)
    if total_seconds is None:
        total_seconds = parse_time_to_seconds(time_s)
    if total_seconds is None and pd.notna(minute_part):
        total_seconds = float(minute_part) * 60

    split_s = parse_time_to_seconds(split_sec_per_500)
    if total_seconds is None and split_s is not None and distance is not None:
        total_seconds = split_s * (float(distance) / 500)

    return split_total_seconds(total_seconds)


def coerce_erg_positive_number(value):
    if value in (None, ""):
        return MISSING_ERG_NUMERIC_VALUE
    if str(value).strip().upper() == "NA":
        return MISSING_ERG_NUMERIC_VALUE

    value = float(value)
    if value <= 0:
        return MISSING_ERG_NUMERIC_VALUE
    return value


def parse_upload_date(value):
    value = clean_value(value)
    if value is None:
        return None

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def athlete_lookup_from_options(options):
    by_name = {}
    by_id = set()
    loose = {}

    for opt in options or []:
        label = str(opt.get("label", "")).strip()
        value = str(opt.get("value", "")).strip()
        if label and value:
            by_name[normalize_person_name(label)] = value
            loose.setdefault(loose_person_name(label), set()).add(value)
        if value:
            by_id.add(value)

    # A punctuation-blind key only earns a place when exactly one athlete owns
    # it. Where two would collide the key is dropped, so the upload reports an
    # unmatched name rather than quietly filing a test under the wrong athlete.
    for key, values in loose.items():
        if len(values) == 1 and key not in by_name:
            by_name[key] = next(iter(values))

    return by_name, by_id


def uploaded_name_lookup_keys(name):
    name = str(name or "").strip()
    if not name:
        return []

    keys = [normalize_person_name(name)]
    if "," in name:
        last, first = [part.strip() for part in name.split(",", 1)]
        if first and last:
            keys.append(normalize_person_name(f"{first} {last}"))

    # Appended last so an exact match is always tried before a looser one.
    keys.extend(loose_person_name(key) for key in list(keys))

    return [key for key in dict.fromkeys(keys) if key]


def resolve_uploaded_profile_id(raw, by_name, unresolved_names):
    profile_id = clean_value(raw.get("profile_id"))
    athlete_name = clean_value(raw.get("athlete"))

    if profile_id is not None:
        profile_id = str(profile_id).strip()
        if profile_id.endswith(".0"):
            profile_id = profile_id[:-2]
        return profile_id

    if athlete_name is not None:
        profile_id = None
        for key in uploaded_name_lookup_keys(athlete_name):
            profile_id = by_name.get(key)
            if profile_id is not None:
                break
        if profile_id is None:
            unresolved_names.append(str(athlete_name))
        return profile_id

    return None


def erg_missing_fields(row):
    """The labels of the fields a row still needs before it can be pushed.

    A power-only test (peak power) needs its watts and nothing else; every other
    piece needs a distance and a time.
    """
    missing = [
        ERG_FIELD_LABELS[field]
        for field in ("profile_id", "test_date", "protocol")
        if row.get(field) in (None, "")
    ]

    if normalize_protocol(row.get("protocol")) in POWER_ONLY_PROTOCOLS:
        if pd.isna(pd.to_numeric(row.get("power_w"), errors="coerce")):
            missing.append(ERG_FIELD_LABELS["power_w"])
        return missing

    if row.get("distance_m") in (None, ""):
        missing.append(ERG_FIELD_LABELS["distance_m"])
    if row.get("time_min") in (None, "") and row.get("time_s") in (None, ""):
        missing.append("time (min or s)")
    return missing


def finalize_erg_upload_rows(rows, skipped_names=None):
    incomplete = 0
    for i, row in enumerate(rows, start=1):
        row["row_no"] = i
        if erg_missing_fields(row):
            incomplete += 1
    return rows, incomplete, sorted(set(skipped_names or []))


ERG_DATA_FIELDS = (
    "profile_id",
    "protocol",
    "distance_m",
    "stroke_rate_spm",
    "power_w",
    "time_min",
    "time_s",
)

ERG_FIELD_LABELS = {
    "profile_id": "athlete",
    "test_date": "test date",
    "protocol": "test",
    "distance_m": "distance (m)",
    "stroke_rate_spm": "stroke rate",
    "power_w": "power",
    "time_min": "time (min)",
    "time_s": "time (s)",
}


# A row the practitioner has started but not finished is the one thing the
# push refuses; tinting the offending cell says so before they press the
# button. A row where nothing at all is filled in is ignored on push, so it is
# left alone here too.
#
# Written as "not every field is blank" rather than the more obvious "any field
# is filled": the DataTable filter grammar has no "is not blank" operator, and
# its one negation ("!") is only accepted at the start of a query or straight
# after a logical operator -- never after an opening bracket.
ERG_ROW_STARTED = "!(" + " && ".join(
    "{%s} is blank" % field for field in ERG_DATA_FIELDS
) + ")"


ERG_POWER_ONLY = '{protocol} = "%s"' % PROTOCOL_PEAK_POWER
ERG_NOT_POWER_ONLY = '{protocol} != "%s"' % PROTOCOL_PEAK_POWER


def erg_missing_cell_styles():
    styles = []
    for field in ("profile_id", "test_date", "protocol"):
        styles.append(
            {
                "if": {
                    "filter_query": "%s && {%s} is blank" % (ERG_ROW_STARTED, field),
                    "column_id": field,
                },
                "backgroundColor": "#fdecea",
                "border": "1px solid #f5c2c7",
            }
        )

    # Distance and time are what every piece but peak power is measured by;
    # peak power is measured by its watts alone.
    styles.append(
        {
            "if": {
                "filter_query": "%s && %s && {distance_m} is blank"
                % (ERG_ROW_STARTED, ERG_NOT_POWER_ONLY),
                "column_id": "distance_m",
            },
            "backgroundColor": "#fdecea",
            "border": "1px solid #f5c2c7",
        }
    )

    for field in ("time_min", "time_s"):
        styles.append(
            {
                "if": {
                    "filter_query": (
                        "%s && %s && {time_min} is blank && {time_s} is blank"
                        % (ERG_ROW_STARTED, ERG_NOT_POWER_ONLY)
                    ),
                    "column_id": field,
                },
                "backgroundColor": "#fdecea",
                "border": "1px solid #f5c2c7",
            }
        )

    styles.append(
        {
            "if": {
                "filter_query": "%s && {power_w} is blank" % ERG_POWER_ONLY,
                "column_id": "power_w",
            },
            "backgroundColor": "#fdecea",
            "border": "1px solid #f5c2c7",
        }
    )

    return styles


def erg_row_label(record, index):
    """Name a row the way the practitioner sees it in the table."""
    row_no = record.get("row_no")
    if row_no in (None, ""):
        return f"table row {index + 1}"
    return f"row {row_no} (table row {index + 1})"


def erg_wide_column_metric(column_name):
    simplified = simplify_header(column_name)
    match = re.search(r"(2000|6000)merg(.+)", simplified)
    if not match:
        return None, None

    distance = int(match.group(1))
    suffix = re.sub(r"\d+$", "", match.group(2).strip())
    if suffix.startswith("avg"):
        suffix = suffix[3:]
    if suffix.startswith("average"):
        suffix = suffix[7:]

    if any(token in suffix for token in ["strokerate", "rate", "spm"]) or suffix == "sr":
        return distance, "stroke_rate_spm"
    if any(token in suffix for token in ["power", "watt"]) or suffix in {"a", "avg", "average", "p", "watts"}:
        return distance, "power_w"
    if suffix in {"r", "m", "min", "mins", "minute", "minutes"}:
        return distance, "time_min"
    if suffix in {"s", "sec", "secs", "second", "seconds"}:
        return distance, "time_s"
    if any(token in suffix for token in ["time", "score", "result"]) or suffix == "t":
        return distance, "time"
    if any(token in suffix for token in ["seconds", "second", "sec"]) or suffix == "times":
        return distance, "time_s"
    if "split" in suffix or suffix == "split":
        return distance, "split_sec_per_500"

    return distance, None


def detect_erg_wide_columns(columns):
    wide = {}
    for column in columns:
        distance, metric = erg_wide_column_metric(column)
        if distance is None or metric is None:
            continue
        wide.setdefault(distance, {})
        wide[distance].setdefault(metric, column)
    return wide


def build_erg_upload_row(profile_id, test_date, distance, values, protocol=None):
    time_min, time_s = normalize_erg_time_parts(
        time_min=values.get("time_min"),
        time_s=values.get("time_s"),
        time_value=values.get("time"),
        split_sec_per_500=values.get("split_sec_per_500"),
        distance=distance,
    )

    stroke_rate = pd.to_numeric(values.get("stroke_rate_spm"), errors="coerce")
    power = pd.to_numeric(values.get("power_w"), errors="coerce")

    total_seconds = None
    if time_min not in (None, "") or time_s not in (None, ""):
        total_seconds = (float(time_min or 0) * 60) + float(time_s or 0)

    return {
        "row_no": None,
        "profile_id": profile_id or "",
        "test_date": test_date or "",
        "protocol": protocol or infer_protocol(distance, total_seconds),
        "distance_m": distance,
        "stroke_rate_spm": float(stroke_rate) if pd.notna(stroke_rate) else "NA",
        "power_w": float(power) if pd.notna(power) else "NA",
        "time_min": time_min,
        "time_s": time_s,
    }


def apply_erg_protocol_defaults(rows):
    """Fill in whatever the chosen piece prescribes, and report if anything moved.

    A 2k and a 6k prescribe their distance; a 30-minute piece prescribes its
    duration and leaves the distance as the result. Filling that in is the
    difference between picking the test and typing it twice.

    A value already in the cell is left alone unless it is the fixed distance of
    a *different* piece -- that is a leftover from the previous choice, not
    something the practitioner meant, and leaving 2000 sitting under a 6k is the
    one way this could quietly record the wrong test.
    """
    fixed_distances = set(PROTOCOL_FIXED_DISTANCE.values())
    changed = False

    for row in rows or []:
        protocol = normalize_protocol(row.get("protocol"))
        if not protocol:
            continue

        current = pd.to_numeric(row.get("distance_m"), errors="coerce")
        is_blank = pd.isna(current)
        looks_prescribed = (not is_blank) and int(current) in fixed_distances
        distance = PROTOCOL_FIXED_DISTANCE.get(protocol)
        minutes = PROTOCOL_FIXED_TIME_MIN.get(protocol)
        has_time = any(row.get(field) not in (None, "") for field in ("time_min", "time_s"))

        if distance is not None:
            if is_blank or (looks_prescribed and int(current) != distance):
                row["distance_m"] = distance
                changed = True
        elif looks_prescribed and not has_time:
            # Switched off a 2k or 6k onto a piece that prescribes no distance,
            # before anything else was typed: the 2000 sitting there is the old
            # choice, not a result. Clearing it is checked against the time
            # being empty so the rule cannot fire twice -- by the next pass a
            # timed piece has its duration filled in, and a distance the
            # practitioner typed themselves is never touched again.
            row["distance_m"] = None
            changed = True

        if minutes is not None and not has_time:
            row["time_min"] = minutes
            changed = True

    return rows, changed


def parse_wide_erg_upload(df, by_name):
    wide_columns = detect_erg_wide_columns(df.columns)
    if not wide_columns:
        return None

    rows = []
    unresolved_names = []

    rename_map = canonical_erg_upload_columns(df.columns)
    df = df.rename(columns=rename_map)
    if "profile_id" not in df.columns and "athlete" not in df.columns:
        raise ValueError("CSV must include either profile_id or Name/Athlete.")

    for _, raw in df.iterrows():
        profile_id = resolve_uploaded_profile_id(raw, by_name, unresolved_names)
        if not profile_id:
            continue
        test_date = parse_upload_date(raw.get("test_date"))

        for distance, metric_columns in sorted(wide_columns.items()):
            values = {
                metric: raw.get(column)
                for metric, column in metric_columns.items()
            }
            if all(clean_value(value) is None for value in values.values()):
                continue
            rows.append(
                build_erg_upload_row(
                    profile_id, test_date, distance, values, infer_protocol(distance)
                )
            )

    return finalize_erg_upload_rows(rows, unresolved_names)


def parse_erg_upload(contents, filename, athlete_options):
    if not contents:
        raise ValueError("No file content provided.")
    if not filename or not filename.lower().endswith(".csv"):
        raise ValueError("Only CSV uploads are supported.")

    _, encoded = contents.split(",", 1)
    decoded = base64.b64decode(encoded)
    df = pd.read_csv(io.StringIO(decode_csv_bytes(decoded)))
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("The uploaded CSV has no data rows.")

    rename_map = canonical_erg_upload_columns(df.columns)
    df = df.rename(columns=rename_map)

    by_name, _ = athlete_lookup_from_options(athlete_options)
    wide_result = parse_wide_erg_upload(df, by_name)
    if wide_result is not None:
        rows, incomplete, skipped_names = wide_result
        if not rows:
            if skipped_names:
                raise ValueError(
                    "No matched athletes were found. Skipped unmatched names: "
                    + ", ".join(skipped_names)
                )
            raise ValueError("No 2000m or 6000m erg results were found in the uploaded CSV.")
        return rows, incomplete, skipped_names

    if "profile_id" not in df.columns and "athlete" not in df.columns:
        raise ValueError("CSV must include either profile_id or Athlete/Name.")

    rows = []
    unresolved_names = []

    for idx, raw in df.iterrows():
        profile_id = resolve_uploaded_profile_id(raw, by_name, unresolved_names)
        if not profile_id:
            continue

        row_no = pd.to_numeric(raw.get("row_no"), errors="coerce")
        distance = pd.to_numeric(raw.get("distance_m"), errors="coerce")
        protocol = normalize_protocol(raw.get("protocol"))
        stroke_rate = pd.to_numeric(raw.get("stroke_rate_spm"), errors="coerce")
        power = pd.to_numeric(raw.get("power_w"), errors="coerce")
        time_min, time_s = normalize_erg_time_parts(
            time_min=raw.get("time_min"),
            time_s=raw.get("time_s"),
            time_value=raw.get("time"),
            distance=distance if pd.notna(distance) else None,
        )

        total_seconds = None
        if time_min not in (None, "") or time_s not in (None, ""):
            total_seconds = (float(time_min or 0) * 60) + float(time_s or 0)

        rows.append(
            {
                "row_no": int(row_no) if pd.notna(row_no) else len(rows) + 1,
                "profile_id": profile_id or "",
                "test_date": parse_upload_date(raw.get("test_date")) or "",
                "protocol": protocol or infer_protocol(
                    int(distance) if pd.notna(distance) else None, total_seconds
                ),
                "distance_m": int(distance) if pd.notna(distance) else None,
                "stroke_rate_spm": float(stroke_rate) if pd.notna(stroke_rate) else "NA",
                "power_w": float(power) if pd.notna(power) else "NA",
                "time_min": time_min,
                "time_s": time_s,
            }
        )

    if not rows and unresolved_names:
        raise ValueError(
            "No matched athletes were found. Skipped unmatched names: "
            + ", ".join(sorted(set(unresolved_names)))
        )

    return finalize_erg_upload_rows(rows, unresolved_names)


def estimate_split_seconds(power_w):
    p = to_float(power_w)
    if p is None or p <= 0:
        return None
    pace = 500.0 * ((2.8 / p) ** (1.0 / 3.0))
    return round(pace, 2)


def format_split_mmss(split_seconds):
    if split_seconds is None:
        return None
    try:
        total_seconds = float(split_seconds)
    except Exception:
        return None
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:05.2f}"


def _interp_y_at_x(df, x_col, y_col, x_target):
    if df is None or df.empty or x_target is None:
        return None

    d = df.copy()
    d[x_col] = pd.to_numeric(d.get(x_col), errors="coerce")
    d[y_col] = pd.to_numeric(d.get(y_col), errors="coerce")
    d = d.dropna(subset=[x_col, y_col])
    if len(d) < 2:
        return None

    d = d.groupby(x_col, as_index=False)[y_col].mean().sort_values(x_col)
    x = d[x_col].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)

    if x_target <= x.min():
        return float(y[0])
    if x_target >= x.max():
        return float(y[-1])

    return float(np.interp(float(x_target), x, y))


# =========================================================
# BULK STEP TEST UPLOAD
# =========================================================
def athlete_labels_from_options(options):
    """profile_id -> display name, for naming sessions back to the uploader."""
    labels = {}
    for option in options or []:
        try:
            labels[int(option.get("value"))] = str(option.get("label", "")).strip()
        except (TypeError, ValueError):
            continue
    return labels


def resolve_bulk_profile_id(raw, by_name):
    """Return (profile_id, unmatched_text).

    An explicit profile_id wins over the name, so a practitioner whose sheet
    spells an athlete differently from the warehouse has a way through that
    does not involve editing this file.
    """
    profile_id = clean_value(raw.get("profile_id"))
    if profile_id is not None:
        text = str(profile_id).strip()
        if text.endswith(".0"):
            text = text[:-2]
        try:
            return int(text), None
        except ValueError:
            return None, text

    athlete = clean_value(raw.get("athlete"))
    if athlete is None:
        return None, None

    for key in uploaded_name_lookup_keys(athlete):
        match = by_name.get(key)
        if match is not None:
            try:
                return int(match), None
            except (TypeError, ValueError):
                return None, str(athlete)

    return None, str(athlete)


def normalize_step_mode(value):
    """'SUB-MAX', 'sub max', 'Submax' -> 'Submax'.

    Returns (value, understood). Blank is understood and means "not stated",
    which the schema allows; an unrecognised word is not, so the caller can
    name it rather than quietly dropping it.
    """
    value = clean_value(value)
    if value is None:
        return None, True

    key = re.sub(r"[^a-z]", "", str(value).lower())
    if key in STEP_MODE_VALUES:
        return STEP_MODE_VALUES[key], True
    return None, False


def parse_bulk_number(value):
    """Returns (number_or_None, understood). Blank is understood as None."""
    value = clean_value(value)
    if value is None:
        return None, True

    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None, False
    return float(number), True


def read_uploaded_csv(contents, filename):
    """Decode a dcc.Upload payload into a DataFrame."""
    if not contents:
        raise ValueError("No file content provided.")
    if not filename or not filename.lower().endswith(".csv"):
        raise ValueError("Only CSV uploads are supported.")

    _, encoded = contents.split(",", 1)
    decoded = base64.b64decode(encoded)
    return pd.read_csv(io.StringIO(decode_csv_bytes(decoded)))


def parse_step_bulk_rows(df, athlete_options):
    """Read every row of a bulk step-test sheet.

    Returns (rows, problems). Every problem in the file is collected and
    reported together, keyed to the spreadsheet line number: a practitioner
    fixing a sheet wants one list to work through, not one error per upload.
    """
    by_name, _ = athlete_lookup_from_options(athlete_options)
    rows = []
    problems = []
    unresolved = {}

    for raw in df.to_dict("records"):
        line = raw.get("_csv_line")

        profile_id, unmatched = resolve_bulk_profile_id(raw, by_name)
        if unmatched is not None:
            unresolved.setdefault(unmatched, []).append(line)
            continue
        if profile_id is None:
            problems.append(
                f"line {line}: no athlete — fill in the athlete or profile_id column"
            )
            continue

        test_date = parse_upload_date(raw.get("test_date"))
        if test_date is None:
            problems.append(
                f"line {line}: could not read the test date "
                f"{clean_value(raw.get('test_date'))!r} — use YYYY-MM-DD"
            )
            continue

        step_no, understood = parse_bulk_number(raw.get("step_no"))
        if not understood or step_no is None or step_no != int(step_no):
            problems.append(
                f"line {line}: step_no {clean_value(raw.get('step_no'))!r} "
                "must be a whole number"
            )
            continue

        mode, understood = normalize_step_mode(raw.get("mode"))
        if not understood:
            problems.append(
                f"line {line}: mode {clean_value(raw.get('mode'))!r} is not "
                "recognised — use Max or Submax"
            )
            continue

        step_type, understood = normalize_step_mode(raw.get("step_type"))
        if not understood:
            problems.append(
                f"line {line}: step_type {clean_value(raw.get('step_type'))!r} is "
                "not recognised — use Max or Submax"
            )
            continue

        raw_test_type = clean_value(raw.get("test_type"))
        test_type = map_test_type(raw_test_type)
        if raw_test_type is not None and test_type is None:
            problems.append(
                f"line {line}: test_type {raw_test_type!r} is not recognised — use "
                "one of " + ", ".join(STEP_TEST_TYPE_VALUES)
            )
            continue

        numbers = {}
        bad_number = False
        for field, label in STEP_NUMERIC_FIELDS.items():
            number, understood = parse_bulk_number(raw.get(field))
            if not understood:
                problems.append(
                    f"line {line}: {label} {clean_value(raw.get(field))!r} is not a number"
                )
                bad_number = True
                continue
            numbers[field] = number
        if bad_number:
            continue

        raw_rpe = clean_value(raw.get("rpe"))
        rpe = parse_rpe(raw_rpe)
        if raw_rpe is not None and rpe is None:
            problems.append(f"line {line}: rpe {raw_rpe!r} is not a number")
            continue

        rows.append(
            {
                "profile_id": profile_id,
                "test_date": test_date,
                "test_type": test_type,
                # The schema wants a string here, never null.
                "notes": clean_value(raw.get("notes")) or "",
                "mode": mode,
                # An unstated step type follows the test's mode, which is how
                # the single-athlete form fills the column too.
                "step_type": step_type or mode,
                "step_no": int(step_no),
                "rpe": rpe,
                **numbers,
            }
        )

    if unresolved:
        details = []
        for name, lines in sorted(unresolved.items()):
            shown = ", ".join(str(line) for line in lines[:5])
            if len(lines) > 5:
                shown += f", …{len(lines) - 5} more"
            details.append(f"{name!r} (line{'s' if len(lines) > 1 else ''} {shown})")
        problems.append(
            "could not match these athlete name(s) to a profile: "
            + "; ".join(details)
            + " — check the spelling against the athlete dropdown on the Step Test "
            "tab, or put the warehouse profile_id in the profile_id column"
        )

    return rows, problems


def build_step_bulk_records(rows, athlete_options):
    """Group parsed rows into test sessions and render warehouse records.

    session_id/session_ts are derived from the athlete and the test date rather
    than from the clock, so re-uploading a corrected sheet lands on the same
    session identity instead of minting a second copy of the test.
    """
    labels = athlete_labels_from_options(athlete_options)

    sessions = {}
    for row in rows:
        key = (row["profile_id"], row["test_date"], row["test_type"], row["notes"])
        sessions.setdefault(key, []).append(row)

    ordered = sorted(
        sessions.items(),
        key=lambda item: (item[0][1], item[0][0], str(item[0][2]), item[0][3]),
    )

    records = []
    preview = []
    seen_per_athlete_day = {}

    for (profile_id, test_date, test_type, notes), session_rows in ordered:
        day = datetime.strptime(test_date, "%Y-%m-%d")
        day_key = (profile_id, test_date)
        index = seen_per_athlete_day.get(day_key, 0) + 1
        seen_per_athlete_day[day_key] = index

        session_id = f"{profile_id}_{day.strftime('%Y%m%d')}_{index:03d}"
        session_ts = day.replace(hour=12).isoformat(timespec="seconds")

        # One body mass per test: a sheet that records it on the first step row
        # only still gets it onto every step of that session.
        body_mass = next(
            (row["body_mass_kg"] for row in session_rows if row["body_mass_kg"] is not None),
            None,
        )

        session_rows = sorted(session_rows, key=lambda row: row["step_no"])
        for row in session_rows:
            records.append(
                {
                    "profile_id": profile_id,
                    "session_id": session_id,
                    "session_ts": session_ts,
                    "test_date": test_date,
                    "body_mass_kg": body_mass,
                    "test_type": test_type,
                    "mode": row["mode"],
                    "notes": notes,
                    "step_no": row["step_no"],
                    "step_type": row["step_type"],
                    "target_po_w": row["target_power_w"],
                    "actual_po_w": row["actual_power_w"],
                    "hr_bpm": row["heart_rate_bpm"],
                    "lactate_mmol": row["lactate_mmol"],
                    "vo2": row["vo2"],
                    "rate_spm": row["stroke_rate_spm"],
                    # Derived exactly as the single-athlete form derives it, so
                    # bulk and manual entry produce identical records.
                    "split_sec_per_500": estimate_split_seconds(row["actual_power_w"]),
                    "rpe": row["rpe"],
                    "time_s": row["time_s"],
                }
            )

        modes = sorted({row["step_type"] for row in session_rows if row["step_type"]})
        preview.append(
            {
                "athlete": labels.get(profile_id, f"profile {profile_id}"),
                "profile_id": profile_id,
                "test_date": test_date,
                "test_type": test_type or "—",
                "mode": "/".join(modes) if modes else "—",
                "steps": len(session_rows),
                "body_mass_kg": body_mass,
                "session_id": session_id,
            }
        )

    return records, preview


def parse_step_bulk_upload(contents, filename, athlete_options):
    """Parse a bulk step-test CSV into (records, preview, problems).

    Nothing is returned for pushing while any problem stands: a half-ingested
    sheet is far harder to unpick than a rejected one.
    """
    df = read_uploaded_csv(contents, filename)
    df = df.rename(columns=canonical_columns(df.columns, STEP_UPLOAD_COLUMN_ALIASES))

    # Carried alongside the data so every problem can name a line the
    # practitioner can actually find in their spreadsheet (the header is 1).
    data_columns = list(df.columns)
    df["_csv_line"] = range(2, len(df) + 2)
    df = df.dropna(how="all", subset=data_columns)
    if df.empty:
        raise ValueError("The uploaded CSV has no data rows.")

    if "profile_id" not in data_columns and "athlete" not in data_columns:
        raise ValueError(
            "CSV must include an 'athlete' column (or 'profile_id'). "
            "Download the template for the expected layout."
        )

    missing = [c for c in STEP_UPLOAD_REQUIRED_COLUMNS if c not in data_columns]
    if missing:
        raise ValueError(
            "CSV is missing required column(s): "
            + ", ".join(missing)
            + ". Download the template for the expected layout."
        )

    rows, problems = parse_step_bulk_rows(df, athlete_options)
    if problems:
        return [], [], problems
    if not rows:
        raise ValueError("No usable rows were found in the uploaded CSV.")

    records, preview = build_step_bulk_records(rows, athlete_options)
    return records, preview, []


def accepted_columns_help(alias_map, template_columns, required=(), either_of=()):
    """Render the accepted header names straight from the alias table.

    Generated rather than written out, so the help can never describe a format
    the parser stopped accepting.
    """
    header_rows = []
    for target in template_columns:
        aliases = alias_map.get(target, set())
        extras = sorted(alias for alias in aliases if alias != simplify_header(target))

        if target in required:
            need, colour = "required", "danger"
        elif target in either_of:
            need, colour = "one of these", "warning"
        else:
            need, colour = "optional", "secondary"

        header_rows.append(
            html.Tr(
                [
                    html.Td(html.Code(target)),
                    html.Td(dbc.Badge(need, color=colour, pill=True)),
                    html.Td(", ".join(extras) or "—", className="text-muted small"),
                ]
            )
        )

    return dbc.Accordion(
        [
            dbc.AccordionItem(
                [
                    html.P(
                        "Case, spaces, underscores and punctuation are ignored when "
                        "matching headers, so \"Heart Rate\" and \"heart_rate_bpm\" are "
                        "the same column. Extra columns we do not recognise are ignored.",
                        className="text-muted small",
                    ),
                    dbc.Table(
                        [
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th("Column"),
                                        html.Th(""),
                                        html.Th("Also accepted"),
                                    ]
                                )
                            ),
                            html.Tbody(header_rows),
                        ],
                        size="sm",
                        borderless=True,
                        striped=True,
                        className="mb-0",
                    ),
                ],
                title="Accepted column names",
            )
        ],
        start_collapsed=True,
        className="mb-3",
    )


def upload_dropzone(component_id, prompt):
    return dcc.Upload(
        id=component_id,
        children=html.Div(prompt),
        style={
            "width": "100%",
            "height": "72px",
            "lineHeight": "72px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "marginBottom": "8px",
        },
        accept=".csv",
        multiple=False,
    )


def problem_list(filename, problems, limit=40):
    """Show every problem at once, capped so one broken column cannot bury the page."""
    shown = problems[:limit]
    body = [
        html.B(
            f"{filename} was not loaded — fix {len(problems)} issue"
            f"{'s' if len(problems) != 1 else ''} and upload again:"
        ),
        html.Ul([html.Li(problem) for problem in shown], className="mb-0 mt-2"),
    ]
    if len(problems) > limit:
        body.append(
            html.Small(f"…and {len(problems) - limit} more.", className="text-muted")
        )
    return html.Div(body)


# =========================================================
# LAYOUT
# =========================================================

layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.Div("PHYSIOLOGY", className="page-eyebrow"),
                    html.H1("Data Entry", className="mb-1"),
                    html.P(
                        "Enter a test, review the results, and save. Drafts are retained in this tab until you start a new test.",
                        className="text-muted mb-0",
                    ),
                ]
            ),
            className="page-title-header align-items-center mt-4 mb-4",
        ),
        dcc.Store(id="athlete-options-store"),
        dcc.Store(id="form-last-submitted-fingerprint"),

        # Table rows are mirrored here on every edit and restored on load.
        # DataTable's own `persistence` only re-applies a saved edit while the
        # layout's `data` still matches what was there when the edit was first
        # recorded — and three callbacks rewrite this table's `data`, so that
        # match is not something to depend on. An explicit store always wins.
        dcc.Store(id="step-table-draft", storage_type=PERSISTENCE_TYPE),
        dcc.Store(id="erg-table-draft", storage_type=PERSISTENCE_TYPE),
        dcc.Interval(id="entry-page-load", interval=250, n_intervals=0, max_intervals=1),

        dcc.Interval(
            id="auth-keepalive-interval",
            interval=AUTH_KEEPALIVE_INTERVAL_MS,
            n_intervals=0,
        ),

        # Reset clears a whole test, so it asks first.
        dcc.ConfirmDialog(
            id="form-reset-confirm",
            message=(
                "Clear the step test form?\n\n"
                "This removes the selected athlete, the test details, the notes "
                "and every step row. It cannot be undone."
            ),
        ),

        # Status messages are pinned to the viewport instead of sitting at the
        # bottom of a narrow column, where an expired session or a failed
        # submit could go unseen while the practitioner works in the table.
        html.Div(
            [
                dbc.Alert(
                    id="form-auth-status-msg",
                    color="warning",
                    is_open=False,
                    dismissable=True,
                    className="shadow",
                ),
                dbc.Alert(
                    id="form-status-msg",
                    color="success",
                    is_open=False,
                    dismissable=True,
                    className="shadow",
                ),
            ],
            className="entry-status-stack",
        ),

        dbc.Tabs(
            [
                # =====================================================
                # STEP TEST TAB
                # =====================================================
                dbc.Tab(
                    label="Step Test",
                    tab_id="tab-step-test",
                    children=[
                        # ---------- Row 1: the two entry tables ----------
                        dbc.Row(
                            [
                                dbc.Col(
                                    make_card(
                                        "Test details",
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Athlete"),
                                                            dcc.Dropdown(
                                                                id="form-name",
                                                                options=[],
                                                                placeholder="Select athlete",
                                                                value=None,
                                                                persistence=True,
                                                                persistence_type=PERSISTENCE_TYPE,
                                                            ),
                                                        ],
                                                        md=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Body Mass (kg)"),
                                                            dbc.Input(
                                                                id="form-mass",
                                                                type="number",
                                                                min=0,
                                                                step=0.1,
                                                                value=None,
                                                                persistence=True,
                                                                persistence_type=PERSISTENCE_TYPE,
                                                            ),
                                                        ],
                                                        md=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Test Date", html_for="form-test-date"),
                                                            dbc.Input(
                                                                type="date",
                                                                size="sm",
                                                                id="form-test-date",
                                                                value=date.today().isoformat(),
                                                                persistence=True,
                                                                persistence_type=PERSISTENCE_TYPE,
                                                            ),
                                                        ],
                                                        md=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Max HR (bpm)"),
                                                            dbc.Input(
                                                                id="form-max-hr",
                                                                type="number",
                                                                min=100,
                                                                max=240,
                                                                step=1,
                                                                placeholder="optional",
                                                                value=None,
                                                                persistence=True,
                                                                persistence_type=PERSISTENCE_TYPE,
                                                            ),
                                                        ],
                                                        md=3,
                                                    ),
                                                ],
                                                className="g-3",
                                            ),
                                            html.Br(),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Equipment"),
                                                            dbc.RadioItems(
                                                                id="form-status",
                                                                options=[
                                                                    {"label": "Erg C2", "value": "erg_C2"},
                                                                    {"label": "Erg RP3", "value": "erg_RP3"},
                                                                    {"label": "On-Water", "value": "row"},
                                                                    {"label": "Bike", "value": "bike"},
                                                                    {"label": "Other", "value": "other"},
                                                                ],
                                                                value=DEFAULT_TEST_TYPE,
                                                                inline=True,
                                                                persistence=True,
                                                                persistence_type=PERSISTENCE_TYPE,
                                                            ),
                                                        ],
                                                        md=12,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Test intensity"),
                                                            dbc.RadioItems(
                                                                id="form-mode",
                                                                options=[
                                                                    {"label": "Max", "value": "Max"},
                                                                    {"label": "Submax", "value": "Submax"},
                                                                ],
                                                                value=DEFAULT_MODE,
                                                                inline=True,
                                                                persistence=True,
                                                                persistence_type=PERSISTENCE_TYPE,
                                                            ),
                                                        ],
                                                        md=12,
                                                    ),
                                                ],
                                                className="g-3",
                                            ),
                                            # Max erg tests only: entered by hand, not derived from the steps.
                                            html.Div(
                                                id="form-max-results",
                                                style={"display": "none"},
                                                children=[
                                                    html.Br(),
                                                    dbc.Row(
                                                        [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("MAP (W)"),
                                                            dbc.Input(
                                                                id="form-map-w",
                                                                type="number",
                                                                min=100,
                                                                max=1000,
                                                                step=1,
                                                                placeholder="optional",
                                                                value=None,
                                                                persistence=True,
                                                                persistence_type=PERSISTENCE_TYPE,
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("VO2max (L/min)"),
                                                            dbc.Input(
                                                                id="form-vo2max",
                                                                type="number",
                                                                min=1,
                                                                max=10,
                                                                step=0.01,
                                                                placeholder="optional",
                                                                value=None,
                                                                persistence=True,
                                                                persistence_type=PERSISTENCE_TYPE,
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                        ],
                                                        className="g-3",
                                                    ),
                                                ],
                                            ),
                                            html.Br(),
                                            dbc.Label("Notes"),
                                            dbc.Textarea(
                                                id="form-notes",
                                                placeholder="Anything you want to capture...",
                                                value="",
                                                style={"height": "110px"},
                                                persistence=True,
                                                persistence_type=PERSISTENCE_TYPE,
                                            ),
                                            html.Br(),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dbc.Button("Save test", id="form-submit", color="primary", className="w-100"),
                                                        md=4,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Button("New test", id="form-reset", color="secondary", outline=True, className="w-100"),
                                                        md=4,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Button("Download CSV", id="form-download-btn", color="info", outline=True, className="w-100"),
                                                        md=4,
                                                    ),
                                                ],
                                                className="g-2",
                                            ),
                                            dcc.Download(id="form-download-csv"),
                                        ],
                                    ),
                                    md=12,
                                ),

                                dbc.Col(
                                    make_card(
                                        "Test steps",
                                        [
                                            html.Div(
                                                [
                                                    dbc.Button("Add step", id="form-add-row", color="success", size="sm", className="me-2"),
                                                    dbc.Button("Delete selected", id="form-delete-rows", color="danger", size="sm", outline=True),
                                                ],
                                                className="mb-2",
                                            ),
                                            dash_table.DataTable(
                                                id="form-items-table",
                                                data=blank_step_rows(),
                                                columns=step_table_columns(DEFAULT_TEST_TYPE),
                                                style_data_conditional=rate_range_styles(DEFAULT_TEST_TYPE),
                                                editable=True,
                                                row_selectable="multi",
                                                selected_rows=[],
                                                page_action="native",
                                                page_size=8,
                                                style_table={"overflowX": "auto"},
                                                style_cell={"padding": "8px", "fontFamily": "system-ui", "fontSize": 14},
                                                style_header={"fontWeight": "600"},
                                            ),
                                        ],
                                    ),
                                    md=12,
                                ),
                            ],
                            className="g-3",
                        ),

                        # ---------- Row 2: everything below, full width ----------
                        html.Hr(),
                        dcc.Store(id="form-last-payload"),

                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="plot-la-vs-po", config={"displayModeBar": False}), md=6),
                                dbc.Col(dcc.Graph(id="plot-hr-vs-po", config={"displayModeBar": False}), md=6),
                            ],
                            className="g-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(make_card("HR~PO Slope", html.H5(id="hr-fit-slope", className="m-0")), md=3),
                                dbc.Col(make_card("HR~PO Intercept", html.H5(id="hr-fit-intercept", className="m-0")), md=3),
                            ],
                            className="g-2 mt-2",
                        ),

                        html.Hr(),
                        make_card(
                            "HR Training Zones (sub-max tests only)",
                            dash_table.DataTable(
                                id="zones-table",
                                data=ZONES_DEFAULT_ROWS,
                                columns=zones_table_columns(DEFAULT_TEST_TYPE),
                                editable=False,
                                style_table={"overflowX": "auto"},
                                style_cell={"padding": "8px", "fontFamily": "system-ui", "fontSize": 14},
                                style_header={"fontWeight": "600"},
                            ),
                        ),

                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        "Download HR Zone Data",
                                        id="form-download-zones-btn",
                                        color="primary",
                                        outline=True,
                                        className="w-100",
                                    ),
                                    md=4,
                                ),
                            ],
                            className="g-2 mt-2",
                        ),
                        dcc.Download(id="form-download-zones-csv"),
                    ],
                ),
                # =====================================================
                # ERG TEST TAB
                # =====================================================
                dbc.Tab(
                    label="Erg Test",
                    tab_id="tab-erg-test",
                    children=[
                        make_card(
                            "Batch Erg Entry",
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dcc.Upload(
                                                id="erg-upload",
                                                children=html.Div("Drag and drop an erg CSV here, or click to select."),
                                                style={
                                                    "width": "100%",
                                                    "height": "72px",
                                                    "lineHeight": "72px",
                                                    "borderWidth": "1px",
                                                    "borderStyle": "dashed",
                                                    "borderRadius": "5px",
                                                    "textAlign": "center",
                                                    "marginBottom": "8px",
                                                },
                                                accept=".csv",
                                                multiple=False,
                                            ),
                                            md=8,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    "Download CSV Template",
                                                    id="erg-template-btn",
                                                    color="secondary",
                                                    outline=True,
                                                    size="sm",
                                                    className="mb-2 w-100",
                                                ),
                                                html.Small(
                                                    "Upload fills the table for review; save after checking rows.",
                                                    className="text-muted",
                                                ),
                                            ],
                                            md=4,
                                        ),
                                    ],
                                    className="g-2 mb-2",
                                ),
                                dcc.Download(id="erg-template-csv"),
                                dbc.Alert(id="erg-upload-status", color="info", is_open=False, className="mb-3"),
                                accepted_columns_help(
                                    ERG_UPLOAD_COLUMN_ALIASES,
                                    ERG_TEMPLATE_COLUMNS,
                                    required=("test_date", "distance_m"),
                                    either_of=("athlete", "profile_id"),
                                ),
                                html.Small(
                                    "Times are whole minutes plus the leftover seconds — a 7:12.4 2k is "
                                    "time_min 7, time_s 12.4. A single \"time\" column written as 7:12.4 "
                                    "works instead. A \"protocol\" column names the piece (2k, 6k, 30min, peak "
                                    "power — which needs only power_w); "
                                    "leave it out and it is read from the distance and the duration. A wide "
                                    "sheet works too: columns named like \"2000m Erg Power\" or \"6000m Erg "
                                    "Rate\" are split into one row per distance automatically.",
                                    className="text-muted d-block mb-3",
                                ),
                                # One toolbar instead of two stacked rows: the date
                                # control and the row actions act on the same table,
                                # so they belong on the same line, with the
                                # irreversible action (Push) held apart on the right.
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dbc.Label(
                                                    "Test date",
                                                    html_for="erg-date-picker",
                                                    className="erg-toolbar-label",
                                                ),
                                                html.Div(
                                                    [
                                                        dbc.Input(
                                                            type="date",
                                                            size="sm",
                                                            id="erg-date-picker",
                                                            value=date.today().isoformat(),
                                                            persistence=True,
                                                            persistence_type=PERSISTENCE_TYPE,
                                                        ),
                                                        dbc.Button(
                                                            "Apply",
                                                            id="erg-apply-date",
                                                            color="secondary",
                                                            outline=True,
                                                            size="sm",
                                                        ),
                                                    ],
                                                    className="erg-toolbar-group",
                                                ),
                                            ],
                                            className="erg-toolbar-field",
                                        ),
                                        html.Div(
                                            [
                                                dbc.Label("Rows", className="erg-toolbar-label"),
                                                html.Div(
                                                    [
                                                        dbc.Button("Add row", id="erg-add-row", color="success", size="sm"),
                                                        dbc.Button("Delete selected", id="erg-delete-rows", color="danger", size="sm", outline=True),
                                                    ],
                                                    className="erg-toolbar-group",
                                                ),
                                            ],
                                            className="erg-toolbar-field",
                                        ),
                                        html.Div(
                                            [
                                                dbc.Label("Save", className="erg-toolbar-label"),
                                                html.Div(
                                                    [
                                                        dbc.Button("Download CSV", id="erg-download-btn", color="info", size="sm", outline=True),
                                                        dbc.Button("Save tests", id="erg-submit", color="primary", size="sm"),
                                                    ],
                                                    className="erg-toolbar-group",
                                                ),
                                            ],
                                            className="erg-toolbar-field ms-auto",
                                        ),
                                    ],
                                    className="erg-toolbar erg-date-controls mb-2",
                                ),
                                html.Small(
                                    "Apply and Delete act on the selected rows \u2014 with nothing "
                                    "selected, Apply sets the date on every row. Cells tinted red are "
                                    "required before that row can be pushed; fully blank rows are ignored.",
                                    className="text-muted d-block mb-3",
                                ),

                                html.Div(
                                    dash_table.DataTable(
                                        id="erg-items-table",
                                        data=blank_erg_rows(),
                                        columns=ERG_TABLE_COLUMNS,
                                        dropdown={
                                            "profile_id": {
                                                "clearable": True,
                                                "options": [],
                                            },
                                            "protocol": {
                                                "clearable": False,
                                                "options": ERG_PROTOCOL_OPTIONS,
                                            },
                                        },
                                        editable=True,
                                        row_selectable="multi",
                                        selected_rows=[],
                                        page_action="native",
                                        page_size=12,
                                        style_table={
                                            "overflowX": "auto",
                                            "overflowY": "visible",
                                            "position": "relative",
                                            "zIndex": 2,
                                        },
                                        style_cell={
                                            "padding": "6px 10px",
                                            "fontFamily": "system-ui",
                                            "fontSize": 14,
                                            "textAlign": "center",
                                            "verticalAlign": "middle",
                                            "height": "48px",
                                        },
                                        style_header={
                                            "fontWeight": "700",
                                            "textAlign": "center",
                                            "backgroundColor": "#f8f9fa",
                                            "border": "1px solid #dee2e6",
                                        },
                                        style_data={
                                            "backgroundColor": "white",
                                            "border": "1px solid #dee2e6",
                                        },
                                        style_data_conditional=[
                                            {
                                                "if": {"column_id": "profile_id"},
                                                "backgroundColor": "#fbfdff",
                                            },
                                            {
                                                "if": {"column_id": "protocol"},
                                                "backgroundColor": "#fbfdff",
                                            },
                                            {
                                                "if": {"row_index": "odd"},
                                                "backgroundColor": "#fcfcfd",
                                            },
                                            # Missing-value tints come last so they
                                            # win over the banding and the dropdown
                                            # column shading above.
                                            *erg_missing_cell_styles(),
                                        ],
                                        style_cell_conditional=[
                                            {"if": {"column_id": "row_no"}, "width": "60px", "color": "#6c757d"},
                                            {"if": {"column_id": "test_date"}, "width": "120px"},
                                            {"if": {"column_id": "protocol"}, "width": "120px"},
                                            {"if": {"column_id": "profile_id"}, "width": "280px", "minWidth": "240px", "textAlign": "left"},
                                            {"if": {"column_id": "distance_m"}, "width": "120px"},
                                            {"if": {"column_id": "stroke_rate_spm"}, "width": "120px"},
                                            {"if": {"column_id": "power_w"}, "width": "110px"},
                                            {"if": {"column_id": "time_min"}, "width": "110px"},
                                            {"if": {"column_id": "time_s"}, "width": "110px"},
                                        ],
                                        tooltip_header={
                                            "profile_id": "Required. Pick from the athlete list.",
                                            "protocol": "Which piece this is. 2000 m and 6000 m fill in the distance; 30 min fills in the time and you enter the distance rowed. Peak power needs only the power.",
                                            "distance_m": "Required (except peak power). Prescribed for a 2k or 6k, the distance covered for a 30 min piece.",
                                            "time_min": "Whole minutes only \u2014 7:12.4 is 7 here.",
                                            "time_s": "Leftover seconds \u2014 7:12.4 is 12.4 here.",
                                        },
                                        tooltip_delay=400,
                                        tooltip_duration=None,
                                        css=[
                                            {
                                                "selector": ".dash-spreadsheet-container .Select-menu-outer",
                                                "rule": """
                                                    display: block !important;
                                                    z-index: 7000 !important;
                                                    max-height: 320px !important;
                                                    border: 1px solid #adb5bd !important;
                                                    border-radius: 8px !important;
                                                    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16) !important;
                                                    overflow-y: auto !important;
                                                """,
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container .Select-option",
                                                "rule": """
                                                    color: #212529 !important;
                                                    background-color: white !important;
                                                    padding: 10px 12px !important;
                                                    line-height: 1.25 !important;
                                                    white-space: normal !important;
                                                """,
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container .Select-option.is-focused",
                                                "rule": "background-color: #eaf3ff !important;",
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container .Select-option.is-selected",
                                                "rule": "background-color: #d7ebff !important;",
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container .Select-value-label",
                                                "rule": """
                                                    color: #212529 !important;
                                                    display: inline-block !important;
                                                    max-width: 100% !important;
                                                    overflow: hidden !important;
                                                    text-overflow: ellipsis !important;
                                                    white-space: nowrap !important;
                                                    line-height: 38px !important;
                                                    padding-left: 4px !important;
                                                    padding-right: 20px !important;
                                                """,
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container .Select-control",
                                                "rule": """
                                                    background-color: transparent !important;
                                                    border: none !important;
                                                    border-radius: 0 !important;
                                                    box-shadow: none !important;
                                                    width: 100% !important;
                                                    min-width: 100% !important;
                                                    max-width: 100% !important;
                                                    min-height: 38px !important;
                                                    height: 38px !important;
                                                    cursor: pointer !important;
                                                """,
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container .Select",
                                                "rule": """
                                                    width: 100% !important;
                                                    min-width: 100% !important;
                                                    max-width: 100% !important;
                                                """,
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container .Select-placeholder",
                                                "rule": """
                                                    color: #6c757d !important;
                                                    line-height: 38px !important;
                                                    padding-left: 4px !important;
                                                    padding-right: 20px !important;
                                                    overflow: hidden !important;
                                                    text-overflow: ellipsis !important;
                                                    white-space: nowrap !important;
                                                """,
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container .Select-input",
                                                "rule": """
                                                    height: 36px !important;
                                                    margin-left: 4px !important;
                                                    padding-left: 0 !important;
                                                """,
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container .Select-arrow-zone",
                                                "rule": """
                                                    padding-right: 6px !important;
                                                    width: 26px !important;
                                                """,
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container .is-focused:not(.is-open) > .Select-control",
                                                "rule": """
                                                    background-color: #f8fbff !important;
                                                    box-shadow: inset 0 0 0 1px #86b7fe !important;
                                                """,
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container .cell--selected",
                                                "rule": "box-shadow: inset 0 0 0 2px #dc3545 !important;",
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container td.focused",
                                                "rule": "box-shadow: inset 0 0 0 2px #dc3545 !important;",
                                            },
                                        ],
                                    ),
                                    className="erg-table-wrap",
                                ),

                                dcc.Download(id="erg-download-csv"),
                                dbc.Alert(id="erg-submit-status", color="success", is_open=False, className="mt-3"),

                                dbc.Row(
                                    [
                                        dbc.Col(make_card("Rows", html.H4(id="erg-row-count", className="m-0")), md=3),
                                        dbc.Col(make_card("Avg Rate", html.H4(id="erg-avg-rate", className="m-0")), md=3),
                                        dbc.Col(make_card("Total Time", html.H4(id="erg-total-time", className="m-0")), md=3),
                                    ],
                                    className="g-2 mt-4",
                                ),
                            ],
                        )
                    ],
                ),
                # =====================================================
                # BULK STEP TEST UPLOAD TAB
                # =====================================================
                dbc.Tab(
                    label="Bulk Step Upload",
                    tab_id="tab-bulk-step",
                    children=[
                        make_card(
                            "Bulk Step Test Upload",
                            [
                                dcc.Store(id="bulk-step-records"),
                                dcc.Store(id="bulk-step-fingerprint"),
                                dcc.Download(id="bulk-step-template-csv"),
                                html.P(
                                    "Upload one CSV holding many athletes' step tests. Use one row "
                                    "per step, and repeat the athlete, date, body mass, test type, "
                                    "mode and notes on every step row of that test. Rows are grouped "
                                    "into sessions by athlete, date, test type and notes.",
                                    className="text-muted",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            upload_dropzone(
                                                "bulk-step-upload",
                                                "Drag and drop a step test CSV here, or click to select.",
                                            ),
                                            md=8,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    "Download CSV Template",
                                                    id="bulk-step-template-btn",
                                                    color="secondary",
                                                    outline=True,
                                                    size="sm",
                                                    className="mb-2 w-100",
                                                ),
                                                html.Small(
                                                    "Nothing is sent until you have read the preview "
                                                    "and pressed Save tests.",
                                                    className="text-muted",
                                                ),
                                            ],
                                            md=4,
                                        ),
                                    ],
                                    className="g-2 mb-2",
                                ),
                                dbc.Alert(
                                    id="bulk-step-status",
                                    color="info",
                                    is_open=False,
                                    className="mb-3",
                                ),
                                accepted_columns_help(
                                    STEP_UPLOAD_COLUMN_ALIASES,
                                    STEP_TEMPLATE_COLUMNS,
                                    required=STEP_UPLOAD_REQUIRED_COLUMNS,
                                    either_of=("athlete", "profile_id"),
                                ),
                                html.H6("Sessions found", className="mt-2"),
                                dash_table.DataTable(
                                    id="bulk-step-preview",
                                    data=[],
                                    columns=BULK_STEP_PREVIEW_COLUMNS,
                                    editable=False,
                                    page_action="native",
                                    page_size=15,
                                    sort_action="native",
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "padding": "6px 10px",
                                        "fontFamily": "system-ui",
                                        "fontSize": 14,
                                        "textAlign": "center",
                                    },
                                    style_cell_conditional=[
                                        {"if": {"column_id": "athlete"}, "textAlign": "left", "minWidth": "200px"},
                                        {"if": {"column_id": "session_id"}, "textAlign": "left"},
                                    ],
                                    style_header={
                                        "fontWeight": "700",
                                        "backgroundColor": "#f8f9fa",
                                    },
                                ),
                                dbc.Button(
                                    "Save tests",
                                    id="bulk-step-submit",
                                    color="primary",
                                    disabled=True,
                                    className="mt-3",
                                ),
                                dbc.Alert(
                                    id="bulk-step-submit-status",
                                    color="success",
                                    is_open=False,
                                    className="mt-3",
                                ),
                            ],
                        )
                    ],
                ),
            ],
            id="entry-tabs",
            active_tab="tab-step-test",
        ),
    ],
    fluid=True,
    className="physiology-page entry-page",
)


# Arrange the existing controls in workflow order: setup, steps, save, results.
_step_tab = layout.children[-1].children[0]
_setup_body = _step_tab.children[0].children[0].children.children[1]
_fields = _setup_body.children
_notes_index = next(i for i, c in enumerate(_fields) if getattr(c, "id", None) == "form-notes")
_actions = _fields[-2:]
_setup_body.children = _fields[:_notes_index - 1] + [
    html.Details([html.Summary("Notes"), _fields[_notes_index]])]
_step_body = _step_tab.children[0].children[1].children.children[1]
_step_body.children += [html.Hr(), *_actions]
_fit_row = _step_tab.children[4]
_step_tab.children[4] = html.Details([html.Summary("Fit details"), _fit_row])


# =========================================================
# ATHLETE OPTIONS
# =========================================================
@dash.callback(
    Output("athlete-options-store", "data"),
    Input("entry-tabs", "id"),   # just something stable so it runs on page load
)
def load_athlete_options(_):
    try:
        token = auth.get_token()
    except Exception:
        raise PreventUpdate

    filters = {"sport_org_id": SPORT_ORG_ID}
    names = fetch_profiles(token, filters)

    return [
        {
            "label": f"{p['person']['first_name']} {p['person']['last_name']}",
            "value": int(p["id"]),
        }
        for p in names
    ]


@dash.callback(
    Output("form-auth-status-msg", "children"),
    Output("form-auth-status-msg", "color"),
    Output("form-auth-status-msg", "is_open"),
    Input("auth-keepalive-interval", "n_intervals"),
)
def keep_auth_session_alive(n_intervals):
    try:
        auth.get_token()
    except Exception as e:
        if is_auth_error(e):
            return auth_relogin_message("submit"), "warning", True
        return "Unable to refresh the login session. Submit may fail if the session has expired.", "warning", True

    return "", "success", False


@dash.callback(
    Output("form-name", "options"),
    Output("erg-items-table", "dropdown"),
    Input("athlete-options-store", "data"),
)
def apply_athlete_options(options):
    if not options:
        return [], {
            "profile_id": {
                "clearable": True,
                "options": [],
            },
            "protocol": {
                "clearable": False,
                "options": ERG_PROTOCOL_OPTIONS,
            },
        }

    erg_options = [
        {"label": opt["label"], "value": str(opt["value"])}
        for opt in options
    ]

    return options, {
        "profile_id": {
            "clearable": True,
            "options": erg_options,
        },
        "protocol": {
            "clearable": False,
            "options": ERG_PROTOCOL_OPTIONS,
        },
    }
# =========================================================
# STEP TEST CALLBACKS
# =========================================================
@dash.callback(
    Output("form-items-table", "data"),
    Output("form-items-table", "selected_rows"),
    Input("form-add-row", "n_clicks"),
    Input("form-delete-rows", "n_clicks"),
    State("form-items-table", "data"),
    State("form-items-table", "selected_rows"),
    State("form-mode", "value"),
    prevent_initial_call=True,
)
def modify_table(add_clicks, del_clicks, rows, selected_rows, mode):
    rows = rows or []
    selected_rows = selected_rows or []
    mode = mode or "Submax"

    if ctx.triggered_id == "form-add-row":
        rows.append({
            "step_no": None,
            "Type": mode,
            "T_PO": None,
            "A_PO": None,
            "HR": None,
            "La": None,
            "V02": None,
            "rate": None,
            "split": None,
            "rpe": None,
            "time_s": None,
        })
        return rows, []

    if ctx.triggered_id == "form-delete-rows":
        if not selected_rows:
            return no_update, no_update
        keep = [r for i, r in enumerate(rows) if i not in set(selected_rows)]
        return keep, []

    return no_update, no_update


@dash.callback(
    Output("form-items-table", "data", allow_duplicate=True),
    Input("form-mode", "value"),
    State("form-items-table", "data"),
    prevent_initial_call=True,
)
def apply_mode_to_all_rows(mode, rows):
    rows = rows or []
    mode = mode or "Submax"
    for r in rows:
        r["Type"] = mode
    return rows


@dash.callback(
    Output("form-items-table", "columns"),
    Output("form-items-table", "style_data_conditional"),
    Output("zones-table", "columns"),
    Input("form-status", "value"),
)
def apply_test_type_columns(test_type):
    """Relabel and re-bound the step table for the test being entered."""
    return (
        step_table_columns(test_type),
        rate_range_styles(test_type),
        zones_table_columns(test_type),
    )


@dash.callback(
    Output("step-table-draft", "data"),
    Input("form-items-table", "data"),
    prevent_initial_call=True,
)
def save_step_draft(rows):
    """Mirror every step-table edit into session storage."""
    return rows


@dash.callback(
    Output("erg-items-table", "data", allow_duplicate=True),
    Input("erg-items-table", "data"),
    prevent_initial_call=True,
)
def fill_erg_protocol_defaults(rows):
    """Picking a piece fills in what that piece prescribes.

    This writes back to the table it watches, so it only returns when something
    actually moved -- the fill is idempotent, and a no-op update would bounce
    between this callback and the draft store forever.
    """
    filled, changed = apply_erg_protocol_defaults([dict(r) for r in (rows or [])])
    if not changed:
        raise PreventUpdate
    return filled


@dash.callback(
    Output("erg-table-draft", "data"),
    Input("erg-items-table", "data"),
    prevent_initial_call=True,
)
def save_erg_draft(rows):
    return rows


@dash.callback(
    Output("form-items-table", "data", allow_duplicate=True),
    Output("erg-items-table", "data", allow_duplicate=True),
    Input("entry-page-load", "n_intervals"),
    State("step-table-draft", "data"),
    State("erg-table-draft", "data"),
    prevent_initial_call=True,
)
def restore_table_drafts(_, step_draft, erg_draft):
    """Put the tables back after a refresh, a navbar mis-click, or the
    token-expiry redirect."""
    step = step_draft if draft_is_restorable(step_draft) else no_update
    erg = erg_draft if draft_is_restorable(erg_draft) else no_update

    if step is no_update and erg is no_update:
        raise PreventUpdate

    return step, erg


@dash.callback(
    Output("form-items-table", "data", allow_duplicate=True),
    Input("form-items-table", "data_timestamp"),
    Input("form-status", "value"),
    State("form-items-table", "data"),
    prevent_initial_call=True,
)
def compute_split_column(_, test_type, rows):
    rows = rows or []
    changed = False
    has_split = step_profile(test_type)["has_split"]

    for r in rows:
        new_split = estimate_split_seconds(r.get("A_PO")) if has_split else None
        if r.get("split") != new_split:
            r["split"] = new_split
            changed = True

    return rows if changed else no_update


@dash.callback(
    Output("form-reset-confirm", "displayed"),
    Input("form-reset", "n_clicks"),
    prevent_initial_call=True,
)
def ask_before_reset(reset_clicks):
    """Reset now really clears the form, so make the practitioner confirm."""
    if not reset_clicks:
        raise PreventUpdate
    return True


@dash.callback(
    Output("form-max-results", "style"),
    Input("form-status", "value"),
    Input("form-mode", "value"),
)
def toggle_max_results(test_type, mode):
    return {"display": "block"} if takes_manual_max_results(test_type, mode) else {"display": "none"}


@dash.callback(
    Output("form-name", "value"),
    Output("form-mass", "value"),
    Output("form-test-date", "value"),
    Output("form-max-hr", "value"),
    Output("form-status", "value"),
    Output("form-mode", "value"),
    Output("form-notes", "value"),
    Output("form-map-w", "value"),
    Output("form-vo2max", "value"),
    Output("form-items-table", "data", allow_duplicate=True),
    Output("form-items-table", "selected_rows", allow_duplicate=True),
    Output("form-last-payload", "data", allow_duplicate=True),
    Output("form-last-submitted-fingerprint", "data", allow_duplicate=True),
    Output("form-status-msg", "children", allow_duplicate=True),
    Output("form-status-msg", "color", allow_duplicate=True),
    Output("form-status-msg", "is_open", allow_duplicate=True),
    Input("form-reset-confirm", "submit_n_clicks"),
    prevent_initial_call=True,
)
def reset_form(confirm_clicks):
    """Actually clear every field. Previously this only wrote a store nobody
    read, so the form stayed populated while reporting success."""
    if not confirm_clicks:
        raise PreventUpdate

    return (
        None,                       # athlete
        None,                       # body mass
        date.today().isoformat(),   # test date
        None,                       # max HR
        DEFAULT_TEST_TYPE,
        DEFAULT_MODE,
        "",                         # notes
        None,                       # MAP
        None,                       # VO2max
        blank_step_rows(),
        [],
        None,                       # last payload
        None,                       # clear the duplicate-submit guard
        "Form cleared.",
        "info",
        True,
    )


@dash.callback(
    Output("form-last-payload", "data"),
    Output("form-last-submitted-fingerprint", "data"),
    Output("form-status-msg", "children"),
    Output("form-status-msg", "color"),
    Output("form-status-msg", "is_open"),
    Input("form-submit", "n_clicks"),
    State("form-name", "value"),
    State("form-mass", "value"),
    State("form-test-date", "value"),
    State("form-status", "value"),
    State("form-mode", "value"),
    State("form-notes", "value"),
    State("form-map-w", "value"),
    State("form-vo2max", "value"),
    State("form-items-table", "data"),
    State("form-last-submitted-fingerprint", "data"),
    prevent_initial_call=True,
    # Disables the button and relabels it for the duration of the ingest, so a
    # slow warehouse call cannot be clicked a second time.
    running=[
        (Output("form-submit", "disabled"), True, False),
        (Output("form-submit", "children"), "Saving…", "Save test"),
        (Output("form-reset", "disabled"), True, False),
    ],
)
def submit_form(
    submit_clicks,
    profile_id,
    mass,
    test_date,
    test_type,
    mode,
    notes,
    map_w,
    vo2max,
    table_rows,
    last_submission,
):
    if not submit_clicks:
        raise PreventUpdate

    # Only a Max erg test carries MAP / VO2max. The fields keep their value
    # while hidden, so drop them here rather than trust what is on screen.
    manual_max = {}
    if takes_manual_max_results(test_type, mode):
        manual_max = {k: to_float(v) for k, v in (("map_w", map_w), ("vo2max", vo2max))}
        issues = manual_max_issues(manual_max)
        if issues:
            return no_update, no_update, " ".join(issues), "warning", True
        # Omit blanks: the fields are new to the schema, and leaving them out
        # keeps a test without them valid either way.
        manual_max = {k: v for k, v in manual_max.items() if v is not None}

    if profile_id is None:
        return no_update, no_update, "Please select an athlete before submitting.", "warning", True

    if test_date is None:
        return no_update, no_update, "Please select a test date before submitting.", "warning", True

    table_rows = table_rows or []
    if not isinstance(table_rows, list) or len(table_rows) == 0:
        return no_update, no_update, "No step data found. Add at least one row.", "warning", True

    session_ts = datetime.now().isoformat(timespec="seconds")
    session_id = f"{int(profile_id)}_{session_ts}"

    records = []
    for r in table_rows:
        has_any = any((r.get(k) not in (None, "", [])) for k in ["T_PO", "A_PO", "HR", "La", "V02", "rate", "rpe", "time_s"])
        if not has_any:
            continue

        records.append({
            "profile_id": int(profile_id),
            "session_id": session_id,
            "session_ts": session_ts,
            "test_date": test_date,
            "body_mass_kg": mass,
            "test_type": test_type,
            "mode": mode,
            "notes": (notes or "").strip(),
            "step_no": r.get("step_no"),
            "step_type": r.get("Type"),
            "target_po_w": r.get("T_PO"),
            "actual_po_w": r.get("A_PO"),
            "hr_bpm": r.get("HR"),
            "lactate_mmol": r.get("La"),
            "vo2": r.get("V02"),
            "rate_spm": r.get("rate"),
            "split_sec_per_500": r.get("split"),
            "rpe": r.get("rpe"),
            "time_s": r.get("time_s"),
            **manual_max,
        })

    if not records:
        return no_update, no_update, "All rows were empty — nothing to submit.", "warning", True

    bad_rate_steps = []
    for rec in records:
        rate = to_float(rec.get("rate_spm"))
        if rate is not None and rate_out_of_range(test_type, rate):
            bad_rate_steps.append(str(rec.get("step_no") or "?"))
    if bad_rate_steps:
        return (
            no_update,
            no_update,
            f"{rate_range_message(test_type)} Check step(s): {', '.join(bad_rate_steps)}.",
            "warning",
            True,
        )

    # Guard against pushing the identical test twice. Each click mints a new
    # session_id, so without this nothing downstream could tell the copies apart.
    fingerprint = submission_fingerprint(records)
    last_submission = last_submission or {}
    if last_submission.get("fingerprint") == fingerprint:
        return (
            no_update,
            no_update,
            (
                f"This exact data was already submitted at "
                f"{last_submission.get('submitted_at', 'an earlier time')} "
                f"(dataset {last_submission.get('dataset_uuid', 'unknown')}). "
                "Change a value, or press Reset, before submitting again."
            ),
            "warning",
            True,
        )

    payload = {
        "timestamp": session_ts,
        "form": {
            "profile_id": int(profile_id),
            "mass": mass,
            "test_date": test_date,
            "test_type": test_type,
            "mode": mode,
            "notes": (notes or "").strip(),
            **manual_max,
        },
        "items": table_rows,
        "session_id": session_id,
        "records_preview_count": len(records),
    }

    try:
        if not VO2_STEP_SOURCE_UUID:
            return payload, no_update, "Ingest failed: VO2_STEP_SOURCE_UUID is not set.", "danger", True

        dataset, created = wc.ingest_raw(
            source_uuid=VO2_STEP_SOURCE_UUID,
            records=records,
            subject_field="profile_id",
            validate_client_side=False,
        )
        submitted = {
            "fingerprint": fingerprint,
            "dataset_uuid": dataset["uuid"],
            "submitted_at": session_ts,
        }
        return (
            payload,
            submitted,
            html.Div([html.Span(f"Saved {created} steps · {test_date}. "), dcc.Link("View reports", href="/reports")]),
            "success",
            True,
        )

    except WarehouseClientError as e:
        if is_auth_error(e):
            return payload, no_update, auth_relogin_message("submit again"), "warning", True
        return payload, no_update, f"Ingest failed: {e}", "danger", True
    except Exception as e:
        if is_auth_error(e):
            return payload, no_update, auth_relogin_message("submit again"), "warning", True
        return payload, no_update, f"Ingest failed unexpectedly: {e}", "danger", True


@dash.callback(
    Output("form-download-csv", "data"),
    Input("form-download-btn", "n_clicks"),
    State("form-items-table", "data"),
    prevent_initial_call=True,
)
def download_csv(n_clicks, rows):
    if not n_clicks:
        raise PreventUpdate
    if not isinstance(rows, list):
        raise ValueError("Rows data should be a list of dictionaries")

    df = pd.DataFrame(rows)
    csv_data = df.to_csv(index=False)
    return dict(content=csv_data, filename="step_test_submission.csv", type="text/csv")


@dash.callback(
    Output("plot-la-vs-po", "figure"),
    Output("plot-hr-vs-po", "figure"),
    Output("hr-fit-slope", "children"),
    Output("hr-fit-intercept", "children"),
    Input("form-items-table", "data"),
)
def update_plots(rows):
    rows = rows or []
    df = pd.DataFrame(rows)

    for c in ["A_PO", "La", "HR"]:
        if c not in df.columns:
            df[c] = None

    df["A_PO"] = pd.to_numeric(df["A_PO"], errors="coerce")
    df["La"] = pd.to_numeric(df["La"], errors="coerce")
    df["HR"] = pd.to_numeric(df["HR"], errors="coerce")

    df_la = df.dropna(subset=["A_PO", "La"])
    df_hr = df.dropna(subset=["A_PO", "HR"])

    fig_la = px.scatter(
        df_la,
        x="A_PO",
        y="La",
        labels={"A_PO": "Actual PO (W)", "La": "Blood Lactate"},
        title="Blood Lactate vs Actual PO",
    )
    fig_la.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    fig_la.update_traces(marker=dict(color="blue"))

    # The same fit, and the same threshold powers, the zones table uses.
    curve = fit_lactate_curve(df_la["A_PO"], df_la["La"]) if len(df_la) >= 3 else None
    if curve is not None:
        x_fit = np.linspace(curve["lo"], curve["hi"], 200)
        fig_la.add_trace(
            go.Scatter(x=x_fit, y=curve["poly"](x_fit), mode="lines", name="La Power Fit", line=dict(color="blue"))
        )
        for target, color in ((LA_2, "#2ca02c"), (LA_4, "#d62728")):
            po = power_at_lactate(curve, target)
            if po is None:
                continue
            fig_la.add_trace(
                go.Scatter(
                    x=[po], y=[target], mode="markers+text",
                    marker=dict(size=11, color=color, symbol="diamond"),
                    text=[f"{target:g} mmol: {po:.0f} W"], textposition="top left",
                    name=f"{target:g} mmol threshold",
                )
            )

    fig_hr = px.scatter(
        df_hr,
        x="A_PO",
        y="HR",
        labels={"A_PO": "Actual PO (W)", "HR": "Heart Rate (bpm)"},
        title="Heart Rate vs Actual PO",
    )
    fig_hr.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    fig_hr.update_traces(marker=dict(color="red"))

    slope_txt = "—"
    intercept_txt = "—"

    if len(df_hr) >= 2:
        lr = sp.stats.linregress(df_hr["A_PO"].values, df_hr["HR"].values)
        slope_txt = f"{lr.slope:.4f} bpm/W"
        intercept_txt = f"{lr.intercept:.2f} bpm"

        x_fit = np.linspace(df_hr["A_PO"].min(), df_hr["A_PO"].max(), 100)
        y_fit = lr.intercept + lr.slope * x_fit

        fig_hr.add_trace(
            go.Scatter(
                x=x_fit,
                y=y_fit,
                mode="lines",
                name=f"Linear fit (R²={lr.rvalue**2:.3f})",
                line=dict(color="red"),
            )
        )

    return fig_la, fig_hr, slope_txt, intercept_txt


@dash.callback(
    Output("zones-table", "data"),
    Input("form-items-table", "data"),
    Input("form-max-hr", "value"),
    Input("form-status", "value"),
    Input("form-mode", "value"),
)
def compute_zones(step_rows, max_hr_input, test_type, mode):
    """HR zones from this test, by the shared method in lactate_thresholds.py --
    the same 2 / 4 mmol powers that are marked on the lactate plot."""
    if not is_submax_test(mode):
        return [dict(row, Notes="Training zones are only calculated from sub-max tests.") for row in ZONES_DEFAULT_ROWS]
    has_split = step_profile(test_type)["has_split"]
    df = pd.DataFrame(step_rows or [])
    if df.empty:
        return ZONES_DEFAULT_ROWS

    for c in ["HR", "La", "A_PO", "rate"]:
        if c not in df.columns:
            df[c] = None
        df[c] = pd.to_numeric(df[c], errors="coerce")

    zones = compute_hr_zones(df["A_PO"], df["HR"], df["La"], max_hr=max_hr_input)
    if zones is None:
        return ZONES_DEFAULT_ROWS

    def rate_at_hr(target_hr):
        return _interp_y_at_x(df, "HR", "rate", target_hr)

    return [format_zone_row(z, rate_at_hr, has_split) for z in zones]


def format_zone_row(zone, rate_at_hr, has_split):
    """Round a shared zone row for the table and add split and rate."""
    def rounded(value, digits):
        return round(value, digits) if value is not None else None

    def split(po):
        return format_split_mmss(estimate_split_seconds(po)) if po is not None and has_split else None

    rate_low = rate_at_hr(zone["HR_low"]) if zone["HR_low"] is not None else None
    rate_high = rate_at_hr(zone["HR_high"]) if zone["HR_high"] is not None else None
    return {
        "Zone": zone["Zone"],
        "HR_low": rounded(zone["HR_low"], 0),
        "HR_high": rounded(zone["HR_high"], 0),
        "PO_low": rounded(zone["PO_low"], 1),
        "PO_high": rounded(zone["PO_high"], 1),
        "Split_low": split(zone["PO_low"]),
        "Split_high": split(zone["PO_high"]),
        "Rate_low": rounded(rate_low, 1),
        "Rate_high": rounded(rate_high, 1),
        "Notes": zone["Notes"],
    }


@dash.callback(
    Output("form-download-zones-csv", "data"),
    Input("form-download-zones-btn", "n_clicks"),
    State("zones-table", "data"),
    prevent_initial_call=True,
)
def download_zones_csv(n_clicks, rows):
    if not n_clicks:
        raise PreventUpdate
    if not isinstance(rows, list):
        raise ValueError("Rows data should be a list of dictionaries")
    df = pd.DataFrame(rows)
    return dict(content=df.to_csv(index=False), filename="HR_Training_Zones.csv", type="text/csv")


# =========================================================
# ERG TEST CALLBACKS
# =========================================================
@dash.callback(
    Output("erg-items-table", "data"),
    Output("erg-items-table", "selected_rows"),
    Output("erg-upload-status", "children"),
    Output("erg-upload-status", "color"),
    Output("erg-upload-status", "is_open"),
    Input("erg-upload", "contents"),
    Input("erg-add-row", "n_clicks"),
    Input("erg-delete-rows", "n_clicks"),
    Input("erg-apply-date", "n_clicks"),
    State("erg-items-table", "data"),
    State("erg-items-table", "selected_rows"),
    State("erg-date-picker", "value"),
    State("erg-upload", "filename"),
    State("athlete-options-store", "data"),
    prevent_initial_call=True,
)
def modify_erg_table(
    upload_contents,
    add_clicks,
    del_clicks,
    apply_date_clicks,
    rows,
    selected_rows,
    selected_date,
    upload_filename,
    athlete_options,
):
    rows = rows or []
    selected_rows = selected_rows or []

    if ctx.triggered_id == "erg-upload":
        try:
            uploaded_rows, incomplete, skipped_names = parse_erg_upload(
                upload_contents,
                upload_filename,
                athlete_options,
            )
        except Exception as e:
            return no_update, no_update, f"CSV upload failed: {e}", "danger", True

        message = f"Loaded {len(uploaded_rows)} erg row(s) from {upload_filename}."
        if incomplete:
            message += f" {incomplete} row(s) need review before pushing."
        if skipped_names:
            message += " Skipped unmatched athlete(s): " + ", ".join(skipped_names) + "."
        return uploaded_rows, [], message, "warning" if incomplete or skipped_names else "success", True

    if ctx.triggered_id == "erg-add-row":
        next_no = (max([r.get("row_no") or 0 for r in rows]) + 1) if rows else 1
        new_row = blank_erg_row(next_no)
        # Carry the piece down from the row above: a session is nearly always
        # one protocol, so the practitioner picks it once.
        if rows:
            new_row["protocol"] = rows[-1].get("protocol")
        rows.append(new_row)
        return rows, [], no_update, no_update, no_update

    if ctx.triggered_id == "erg-delete-rows":
        if not selected_rows:
            return no_update, no_update, no_update, no_update, no_update
        keep = [r for i, r in enumerate(rows) if i not in set(selected_rows)]
        return keep, [], no_update, no_update, no_update

    if ctx.triggered_id == "erg-apply-date":
        if not selected_date:
            return no_update, no_update, no_update, no_update, no_update
        indexes = set(selected_rows) if selected_rows else set(range(len(rows)))
        for i, row in enumerate(rows):
            if i in indexes:
                row["test_date"] = selected_date
        return rows, selected_rows, no_update, no_update, no_update

    return no_update, no_update, no_update, no_update, no_update


@dash.callback(
    Output("erg-row-count", "children"),
    Output("erg-avg-rate", "children"),
    Output("erg-total-time", "children"),
    Input("erg-items-table", "data"),
)
def update_erg_summary(rows):
    rows = rows or []
    df = pd.DataFrame(rows)

    if df.empty:
        return "0", "—", "—"

    for c in ["power_w", "stroke_rate_spm", "time_min", "time_s"]:
        if c not in df.columns:
            df[c] = None

    r = pd.to_numeric(df["stroke_rate_spm"], errors="coerce")
    t_s = pd.to_numeric(df["time_s"], errors="coerce")
    t_min = pd.to_numeric(df["time_min"], errors="coerce")
    t = (t_min.fillna(0) * 60) + t_s.fillna(0)
    t = t.where(t_min.notna() | t_s.notna())

    avg_r = r.mean(skipna=True)
    total_t = t.sum(skipna=True)

    avg_r_txt = f"{avg_r:.1f} spm" if pd.notna(avg_r) else "—"
    total_t_txt = f"{total_t / 60:.2f} min" if pd.notna(total_t) else "—"

    return str(len(rows)), avg_r_txt, total_t_txt


@dash.callback(
    Output("erg-download-csv", "data"),
    Input("erg-download-btn", "n_clicks"),
    State("erg-items-table", "data"),
    prevent_initial_call=True,
)
def download_erg_csv(n_clicks, rows):
    if not n_clicks:
        raise PreventUpdate
    if not isinstance(rows, list):
        raise ValueError("Rows data should be a list of dictionaries")
    df = pd.DataFrame(rows)
    return dict(content=df.to_csv(index=False), filename="erg_batch_entry.csv", type="text/csv")


@dash.callback(
    Output("erg-submit-status", "children"),
    Output("erg-submit-status", "color"),
    Output("erg-submit-status", "is_open"),
    Input("erg-submit", "n_clicks"),
    State("erg-items-table", "data"),
    prevent_initial_call=True,
)
def submit_erg_records(n_clicks, rows):
    if not n_clicks:
        raise PreventUpdate

    if not ERG_TEST_SOURCE_UUID:
        return "Push failed: ERG_TEST_SOURCE_UUID is not set in settings.py.", "danger", True

    records = []
    problems = []
    blank_rows = 0

    for index, row in enumerate(rows or []):
        raw_protocol = row.get("protocol")
        record = {
            "row_no": row.get("row_no"),
            "profile_id": row.get("profile_id"),
            "test_date": row.get("test_date"),
            "protocol": normalize_protocol(raw_protocol),
            "distance_m": row.get("distance_m"),
            "stroke_rate_spm": row.get("stroke_rate_spm"),
            "power_w": row.get("power_w"),
            "time_min": row.get("time_min"),
            "time_s": row.get("time_s"),
        }

        # A row counts as blank only when every data field is empty. Anything
        # the practitioner actually typed is validated and reported, never
        # dropped quietly for lacking one of the required fields.
        if all(record[field] in (None, "") for field in ERG_DATA_FIELDS):
            blank_rows += 1
            continue

        label = erg_row_label(record, index)

        # Blank row numbers get filled from the table position instead of
        # blowing up in int() with an unhelpful message.
        if record["row_no"] in (None, ""):
            record["row_no"] = index + 1

        if record["protocol"] is None and raw_protocol not in (None, ""):
            problems.append(
                f"{label} has an unrecognised test \"{raw_protocol}\" — use one of "
                + ", ".join(protocol_label(p) for p in ERG_PROTOCOL_VALUES)
            )
            continue

        missing = erg_missing_fields(record)
        if missing:
            problems.append(f"{label} is missing {', '.join(missing)}")
            continue

        if record["protocol"] not in ERG_PROTOCOL_VALUES:
            problems.append(
                f"{label} has an unknown test — use one of "
                + ", ".join(protocol_label(p) for p in ERG_PROTOCOL_VALUES)
            )
            continue

        try:
            datetime.strptime(str(record["test_date"]), "%Y-%m-%d")
        except (TypeError, ValueError):
            problems.append(f"{label} has an invalid test date — use YYYY-MM-DD")
            continue

        try:
            record["row_no"] = int(record["row_no"])
            record["profile_id"] = int(record["profile_id"])
            power_only = record["protocol"] in POWER_ONLY_PROTOCOLS
            has_time = record["time_min"] not in (None, "") or record["time_s"] not in (None, "")
            if record["distance_m"] in (None, "") and power_only:
                record["distance_m"] = None
            else:
                record["distance_m"] = int(record["distance_m"])
            record["stroke_rate_spm"] = coerce_erg_positive_number(record["stroke_rate_spm"])
            record["power_w"] = coerce_erg_positive_number(record["power_w"])
            if has_time or not power_only:
                record["time_min"], record["time_s"] = normalize_erg_time_parts(
                    time_min=record["time_min"],
                    time_s=record["time_s"],
                )
                if record["time_min"] is None or record["time_s"] is None:
                    raise ValueError
            # A peak power test with no time is stored with the same 0.1
            # placeholder as any other missing number.
            record["time_min"] = coerce_erg_positive_number(record["time_min"])
            record["time_s"] = coerce_erg_positive_number(record["time_s"])
        except (TypeError, ValueError):
            problems.append(f"{label} contains an invalid value — enter time as minutes and/or seconds")
            continue

        records.append(record)

    # Report every bad row at once rather than one per push attempt.
    if problems:
        return (
            "Push failed — nothing was sent. Fix these rows: " + "; ".join(problems) + ".",
            "danger",
            True,
        )

    if not records:
        return "Push failed: enter at least one complete erg result.", "danger", True

    try:
        dataset, created = wc.ingest_raw(
            source_uuid=ERG_TEST_SOURCE_UUID,
            records=records,
            subject_field="profile_id",
            validate_client_side=False,
        )
        message = f"Submitted {created} erg row(s). Dataset UUID: {dataset['uuid']}"
        if blank_rows:
            message += f" ({blank_rows} blank row(s) ignored.)"
        return message, "success", True
    except WarehouseClientError as e:
        if is_auth_error(e):
            return auth_relogin_message("submit again"), "warning", True
        return f"Push failed: {e}", "danger", True
    except Exception as e:
        if is_auth_error(e):
            return auth_relogin_message("submit again"), "warning", True
        return f"Push failed unexpectedly: {e}", "danger", True


# =========================================================
# TEMPLATE DOWNLOADS
# =========================================================
@dash.callback(
    Output("bulk-step-template-csv", "data"),
    Input("bulk-step-template-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_step_bulk_template(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return dict(
        content=step_template_csv(),
        filename="step_test_bulk_template.csv",
        type="text/csv",
    )


@dash.callback(
    Output("erg-template-csv", "data"),
    Input("erg-template-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_erg_template(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return dict(
        content=erg_template_csv(),
        filename="erg_bulk_template.csv",
        type="text/csv",
    )


# =========================================================
# BULK STEP TEST UPLOAD CALLBACKS
# =========================================================
@dash.callback(
    Output("bulk-step-records", "data"),
    Output("bulk-step-preview", "data"),
    Output("bulk-step-status", "children"),
    Output("bulk-step-status", "color"),
    Output("bulk-step-status", "is_open"),
    Output("bulk-step-submit", "disabled"),
    Output("bulk-step-submit", "children"),
    Input("bulk-step-upload", "contents"),
    State("bulk-step-upload", "filename"),
    State("athlete-options-store", "data"),
    prevent_initial_call=True,
)
def load_step_bulk_upload(contents, filename, athlete_options):
    if not contents:
        raise PreventUpdate

    idle = (None, [], True, "Save tests")

    if not athlete_options:
        return (
            *idle[:2],
            "The athlete list has not loaded yet — wait a moment, then upload again.",
            "warning",
            True,
            *idle[2:],
        )

    try:
        records, preview, problems = parse_step_bulk_upload(
            contents, filename, athlete_options
        )
    except Exception as e:
        return (*idle[:2], f"CSV upload failed: {e}", "danger", True, *idle[2:])

    if problems:
        return (
            *idle[:2],
            problem_list(filename, problems),
            "danger",
            True,
            *idle[2:],
        )

    message = (
        f"Loaded {len(records)} step row(s) across {len(preview)} test session(s) "
        f"from {filename}. Check the sessions below, then push."
    )
    return (
        records,
        preview,
        message,
        "success",
        True,
        False,
        f"Push {len(records)} row(s) to Warehouse",
    )


@dash.callback(
    Output("bulk-step-submit-status", "children"),
    Output("bulk-step-submit-status", "color"),
    Output("bulk-step-submit-status", "is_open"),
    Output("bulk-step-fingerprint", "data"),
    Output("bulk-step-submit", "disabled", allow_duplicate=True),
    Input("bulk-step-submit", "n_clicks"),
    State("bulk-step-records", "data"),
    State("bulk-step-fingerprint", "data"),
    prevent_initial_call=True,
)
def push_step_bulk_records(n_clicks, records, last_submission):
    if not n_clicks:
        raise PreventUpdate

    if not records:
        return "Nothing to push — upload a CSV first.", "warning", True, no_update, True

    if not VO2_STEP_SOURCE_UUID:
        return (
            "Push failed: VO2_STEP_SOURCE_UUID is not set in settings.py.",
            "danger",
            True,
            no_update,
            False,
        )

    # session_id is derived from athlete and date, so a second push of the same
    # sheet is indistinguishable downstream from the first. Catch it here.
    fingerprint = submission_fingerprint(records)
    last_submission = last_submission or {}
    if last_submission.get("fingerprint") == fingerprint:
        return (
            (
                "This exact file was already pushed at "
                f"{last_submission.get('submitted_at', 'an earlier time')} "
                f"(dataset {last_submission.get('dataset_uuid', 'unknown')}). "
                "Upload a changed file if you need to push again."
            ),
            "warning",
            True,
            no_update,
            True,
        )

    submitted_at = datetime.now().isoformat(timespec="seconds")
    try:
        dataset, created = wc.ingest_raw(
            source_uuid=VO2_STEP_SOURCE_UUID,
            records=records,
            subject_field="profile_id",
            validate_client_side=False,
        )
    except Exception as e:
        if is_auth_error(e):
            return auth_relogin_message("push again"), "warning", True, no_update, False
        return f"Push failed: {e}", "danger", True, no_update, False

    return (
        f"Pushed {created} step row(s). Dataset UUID: {dataset.get('uuid')}",
        "success",
        True,
        {
            "fingerprint": fingerprint,
            "dataset_uuid": dataset.get("uuid"),
            "submitted_at": submitted_at,
        },
        True,
    )


@dash.callback(
    Output("form-download-zones-btn", "disabled"),
    Input("form-mode", "value"),
)
def disable_max_zone_export(mode):
    return not is_submax_test(mode)
