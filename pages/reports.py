# pages/reporting.py

from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np

import dash
from dash import html, dcc, dash_table, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import plotly.express as px
import plotly.graph_objects as go

from auth_setup import auth
from utils import fetch_profiles
from settings import SITE_URL, VO2_STEP_SOURCE_UUID
from warehouse import WarehouseAPIConfig, WarehouseClient, WarehouseClientError
from step_protocols import rate_out_of_range, rate_range_message

import erg_editor  # registers the Erg Scores tab and its callbacks


cfg = WarehouseAPIConfig(base_url=SITE_URL)
wc = WarehouseClient(cfg, token_getter=auth.get_token)

dash.register_page(__name__, path="/reports", name="Reporting")


# =========================================================
# CONSTANTS / DEFAULTS
# =========================================================
REPORT_TABLE_COLUMNS = [
    {"name": "Record UUID", "id": "__record_uuid", "editable": False},
    {"name": "Row Issues", "id": "__row_issues", "editable": False},
    {"name": "Test Date", "id": "test_date", "editable": False},
    {"name": "Athlete", "id": "athlete_name", "editable": False},
    {"name": "Athlete ID", "id": "profile_id", "editable": False},
    {"name": "Session ID", "id": "session_id", "editable": False},
    {"name": "Step", "id": "step_no", "editable": False},
    {"name": "Mode", "id": "mode", "editable": True, "presentation": "dropdown"},
    {"name": "Test Type", "id": "test_type", "editable": True, "presentation": "dropdown"},
    {"name": "Target PO", "id": "target_po_w", "editable": True},
    {"name": "Actual PO", "id": "actual_po_w", "editable": True},
    {"name": "HR", "id": "hr_bpm", "editable": True},
    {"name": "La", "id": "lactate_mmol", "editable": True},
    {"name": "VO2", "id": "vo2", "editable": True},
    {"name": "Rate", "id": "rate_spm", "editable": True},
    {"name": "Split (s/500)", "id": "split_sec_per_500", "editable": False},
    {"name": "RPE", "id": "rpe", "editable": True},
    {"name": "Time (s)", "id": "time_s", "editable": True},
    {"name": "Body Mass", "id": "body_mass_kg", "editable": True},
    {"name": "Notes", "id": "notes", "editable": True},
]

REPORT_DROPDOWN_OPTIONS = {
    "mode": [
        {"label": "Max", "value": "Max"},
        {"label": "Submax", "value": "Submax"},
        {"label": "Sub Max", "value": "Sub Max"},
        {"label": "max", "value": "max"},
        {"label": "submax", "value": "submax"},
    ],
    "test_type": [
        {"label": "Erg C2", "value": "erg_C2"},
        {"label": "Erg RP3", "value": "erg_RP3"},
        {"label": "On-Water", "value": "row"},
        {"label": "Bike", "value": "bike"},
        {"label": "Other", "value": "other"},
    ],
}

REPORT_EDITABLE_COLUMNS = {
    col["id"]
    for col in REPORT_TABLE_COLUMNS
    if col.get("editable") and not col["id"].startswith("__")
}

REPORT_NUMERIC_COLUMNS = {
    "profile_id",
    "body_mass_kg",
    "step_no",
    "target_po_w",
    "actual_po_w",
    "hr_bpm",
    "lactate_mmol",
    "vo2",
    "rate_spm",
    "split_sec_per_500",
    "rpe",
    "time_s",
}

REPORT_INTEGER_COLUMNS = {
    "profile_id",
    "step_no",
}

REPORT_STRING_DEFAULTS = {
    "notes": "",
}

REPORT_DATA_COLUMNS = [
    "profile_id",
    "session_id",
    "session_ts",
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
]

REPORT_NON_PAYLOAD_COLUMNS = {"athlete_name", "__row_issues"}

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


# =========================================================
# HELPERS
# =========================================================


def make_session_label(df):
    """
    Short display label for legend, while preserving separate sessions internally.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=list(df.columns) + ["test_date_dt", "session_label", "session_key"]) if df is not None else pd.DataFrame(columns=["test_date_dt", "session_label", "session_key"])

    dff = df.copy()

    for col, default in [("test_date", pd.NaT), ("session_id", None), ("mode", None)]:
        if col not in dff.columns:
            dff[col] = default

    dff["test_date_dt"] = pd.to_datetime(dff["test_date"], errors="coerce")

    def _label(row):
        dt = row["test_date_dt"]
        date_txt = dt.strftime("%Y-%m-%d") if pd.notna(dt) else "Unknown date"
        mode_txt = str(row.get("mode", "Unknown")).strip() if pd.notna(row.get("mode", None)) else "Unknown"
        return f"{date_txt} | {mode_txt}"

    def _key(row):
        dt = row["test_date_dt"]
        date_txt = dt.strftime("%Y-%m-%d") if pd.notna(dt) else "Unknown date"
        mode_txt = str(row.get("mode", "Unknown")).strip() if pd.notna(row.get("mode", None)) else "Unknown"
        sid = row.get("session_id", "")
        return f"{date_txt} | {mode_txt} | {sid}"

    dff["session_label"] = dff.apply(_label, axis=1)   # short legend text
    dff["session_key"] = dff.apply(_key, axis=1)       # unique grouping key
    return dff

def estimate_power_at_lactate_thresholds(session_df, thresholds=(2, 4, 6), grid_n=400):
    """
    Estimate the power (W) at given lactate thresholds using a curve fit of
    lactate vs actual power within a single session.

    Approach:
    - Uses quadratic fit if >= 3 unique points
    - Falls back to linear fit if only 2 unique points
    - Estimates threshold power from a dense grid within the observed power range
    - Returns NaN if threshold is outside fitted range
    """
    dff = session_df.dropna(subset=["actual_po_w", "lactate_mmol"]).copy()

    if dff.empty:
        return {f"lt_{int(t)}_w": np.nan for t in thresholds} | {"fit_degree": np.nan, "n_points": 0}

    # collapse duplicate powers to mean lactate
    dff = (
        dff.groupby("actual_po_w", as_index=False)["lactate_mmol"]
        .mean()
        .sort_values("actual_po_w")
    )

    x = dff["actual_po_w"].to_numpy(dtype=float)
    y = dff["lactate_mmol"].to_numpy(dtype=float)

    # need at least 2 unique power points
    if len(x) < 2 or np.nanmin(x) == np.nanmax(x):
        return {f"lt_{int(t)}_w": np.nan for t in thresholds} | {"fit_degree": np.nan, "n_points": len(x)}

    # choose fit degree
    deg = 2 if len(x) >= 3 else 1

    try:
        coeffs = np.polyfit(x, y, deg=deg)
        poly = np.poly1d(coeffs)
    except Exception:
        return {f"lt_{int(t)}_w": np.nan for t in thresholds} | {"fit_degree": np.nan, "n_points": len(x)}

    grid = np.linspace(np.nanmin(x), np.nanmax(x), grid_n)
    yhat = poly(grid)

    out = {}
    yhat_min = np.nanmin(yhat)
    yhat_max = np.nanmax(yhat)

    for t in thresholds:
        col = f"lt_{int(t)}_w"

        # only estimate if threshold is within fitted range
        if t < min(yhat_min, yhat_max) or t > max(yhat_min, yhat_max):
            out[col] = np.nan
            continue

        idx = np.nanargmin(np.abs(yhat - t))
        out[col] = float(grid[idx])

    out["fit_degree"] = deg
    out["n_points"] = len(x)
    return out


def build_lactate_threshold_trend_df(df):
    """
    One row per session with estimated LT2 / LT4 / LT6 powers.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    sess = df.copy()
    sess["test_date_dt"] = pd.to_datetime(sess["test_date"], errors="coerce")

    rows = []
    group_cols = ["session_id", "profile_id", "athlete_name", "test_date_dt"]

    for keys, g in sess.groupby(group_cols, dropna=False):
        session_id, profile_id, athlete_name, test_date_dt = keys

        est = estimate_power_at_lactate_thresholds(g, thresholds=(2, 4, 6))

        rows.append({
            "session_id": session_id,
            "profile_id": profile_id,
            "athlete_name": athlete_name,
            "test_date_dt": test_date_dt,
            "lt_2_w": est["lt_2_w"],
            "lt_4_w": est["lt_4_w"],
            "lt_6_w": est["lt_6_w"],
            "fit_degree": est["fit_degree"],
            "n_points": est["n_points"],
            "max_po": g["actual_po_w"].max(skipna=True),
            "max_la": g["lactate_mmol"].max(skipna=True),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["test_date_dt", "athlete_name", "session_id"])







def make_card(title, body):
    return dbc.Card(
        [dbc.CardHeader(html.B(title)), dbc.CardBody(body)],
        className="shadow-sm",
    )


def to_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def format_split_mmss(split_seconds):
    if split_seconds is None or pd.isna(split_seconds):
        return "—"
    try:
        total_seconds = float(split_seconds)
    except Exception:
        return "—"
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:05.2f}"


def safe_date_str(x):
    if x is None or x == "":
        return None
    try:
        return pd.to_datetime(x).date().isoformat()
    except Exception:
        return None


def estimate_split_seconds(power_w):
    p = to_float(power_w)
    if p is None or p <= 0:
        return None
    pace = 500.0 * ((2.8 / p) ** (1.0 / 3.0))
    return round(pace, 2)


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


def clean_cell_value(value, column_id):
    if column_id in REPORT_STRING_DEFAULTS and (value is None or value == "" or pd.isna(value)):
        return REPORT_STRING_DEFAULTS[column_id]
    if value is None or value == "":
        return None
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if column_id in REPORT_NUMERIC_COLUMNS:
        numeric_value = to_float(value)
        if numeric_value is None:
            return None
        if column_id in REPORT_INTEGER_COLUMNS:
            return int(numeric_value)
        return numeric_value
    if column_id == "test_date":
        return safe_date_str(value)
    if column_id == "session_ts":
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)
    return value


def row_to_warehouse_payload(row):
    payload = {}
    for col in REPORT_DATA_COLUMNS:
        payload[col] = clean_cell_value(row.get(col), col)
    return payload


def values_equal(a, b, column_id):
    a_clean = clean_cell_value(a, column_id)
    b_clean = clean_cell_value(b, column_id)

    if a_clean is None and b_clean is None:
        return True

    if column_id in REPORT_NUMERIC_COLUMNS:
        try:
            return np.isclose(float(a_clean), float(b_clean), equal_nan=True)
        except Exception:
            return a_clean == b_clean

    return a_clean == b_clean


def display_value(value, column_id=None):
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    if column_id == "split_sec_per_500":
        return format_split_mmss(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def row_label(row):
    athlete = row.get("athlete_name") or row.get("profile_id") or "Athlete"
    test_date = display_value(row.get("test_date"))
    session = row.get("session_id") or "No session"
    step = display_value(row.get("step_no"))
    return f"{athlete} | {test_date} | {session} | Step {step}"


def get_changed_cells(original_records, edited_rows):
    original_df = normalize_records_to_df(original_records)
    edited_df = pd.DataFrame(edited_rows or [])

    if original_df.empty or edited_df.empty or "__record_uuid" not in edited_df.columns:
        return []

    original_by_uuid = {
        str(row["__record_uuid"]): row
        for row in original_df.to_dict("records")
        if row.get("__record_uuid")
    }

    changes = []
    for edited_row in edited_df.to_dict("records"):
        record_uuid = edited_row.get("__record_uuid")
        if not record_uuid:
            continue

        original_row = original_by_uuid.get(str(record_uuid))
        if not original_row:
            continue

        for col in REPORT_EDITABLE_COLUMNS:
            if not values_equal(edited_row.get(col), original_row.get(col), col):
                changes.append(
                    {
                        "record_uuid": str(record_uuid),
                        "column_id": col,
                        "column_name": next(
                            (c["name"] for c in REPORT_TABLE_COLUMNS if c["id"] == col),
                            col,
                        ),
                        "old": clean_cell_value(original_row.get(col), col),
                        "new": clean_cell_value(edited_row.get(col), col),
                        "row_label": row_label(edited_row),
                    }
                )

    return changes


def validate_reporting_rows(edited_rows):
    issues = []
    for idx, row in enumerate(edited_rows or []):
        label = row_label(row)
        record_uuid = row.get("__record_uuid")

        def add_issue(column_id, message):
            issues.append(
                {
                    "row_index": idx,
                    "record_uuid": str(record_uuid) if record_uuid else "",
                    "column_id": column_id,
                    "row_label": label,
                    "message": message,
                }
            )

        numeric_ranges = {
            "actual_po_w": (0, 1200, "Actual PO should be between 0 and 1200 W."),
            "target_po_w": (0, 1200, "Target PO should be between 0 and 1200 W."),
            "hr_bpm": (30, 240, "HR should be between 30 and 240 bpm."),
            "lactate_mmol": (0, 30, "Lactate should be between 0 and 30 mmol/L."),
            "vo2": (0, 100, "VO2 should be between 0 and 100."),
            "rpe": (0, 20, "RPE should be between 0 and 20."),
            "time_s": (0, 7200, "Time should be between 0 and 7200 seconds."),
            "body_mass_kg": (20, 250, "Body mass should be between 20 and 250 kg."),
        }

        for col, (low, high, message) in numeric_ranges.items():
            raw_value = row.get(col)
            if raw_value in (None, ""):
                continue
            value = to_float(raw_value)
            if value is None:
                add_issue(col, f"{col} must be numeric.")
            elif value < low or value > high:
                add_issue(col, message)

        # Rate bounds depend on the test: stroke rate on a rower, cadence on a bike.
        rate_value = row.get("rate_spm")
        if rate_value not in (None, ""):
            value = to_float(rate_value)
            if value is None:
                add_issue("rate_spm", "rate_spm must be numeric.")
            elif rate_out_of_range(row.get("test_type"), value):
                add_issue("rate_spm", rate_range_message(row.get("test_type")))

        for col, options in REPORT_DROPDOWN_OPTIONS.items():
            value = row.get(col)
            valid_values = {opt["value"] for opt in options}
            if value not in (None, "") and value not in valid_values:
                add_issue(col, f"{col} should use one of the known options.")

    return issues


def build_change_summary(changes, issues):
    if issues:
        issue_rows = [
            html.Li(f"{issue['row_label']} - {issue['message']}")
            for issue in issues[:12]
        ]
        if len(issues) > 12:
            issue_rows.append(html.Li(f"...and {len(issues) - 12} more issue(s)."))
        return html.Div(
            [
                html.P("Resolve these issues before updating the warehouse."),
                html.Ul(issue_rows, className="mb-0"),
            ]
        )

    if not changes:
        return html.Div("No editable changes detected.")

    grouped = {}
    for change in changes:
        grouped.setdefault(change["row_label"], []).append(change)

    blocks = []
    for label, row_changes in list(grouped.items())[:8]:
        blocks.append(html.H6(label, className="mt-2 mb-1"))
        blocks.append(
            dbc.Table(
                [
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(change["column_name"]),
                                    html.Td(display_value(change["old"], change["column_id"])),
                                    html.Td(display_value(change["new"], change["column_id"])),
                                ]
                            )
                            for change in row_changes
                        ]
                    )
                ],
                bordered=True,
                size="sm",
                className="mb-2",
            )
        )

    if len(grouped) > 8:
        blocks.append(html.Div(f"...and {len(grouped) - 8} more changed row(s)."))

    return html.Div(
        [
            html.P(f"Ready to update {len(grouped)} row(s), {len(changes)} field change(s)."),
            dbc.Table(
                html.Thead(html.Tr([html.Th("Field"), html.Th("Current"), html.Th("New")])),
                bordered=True,
                size="sm",
                className="mb-1",
            ),
            *blocks,
        ]
    )


def build_reporting_table_styles(changes, issues):
    styles = [
        {
            "if": {"column_id": col},
            "backgroundColor": "#f3f9ff",
        }
        for col in REPORT_EDITABLE_COLUMNS
    ]

    for change in changes:
        styles.append(
            {
                "if": {
                    "filter_query": f'{{__record_uuid}} = "{change["record_uuid"]}"',
                    "column_id": change["column_id"],
                },
                "backgroundColor": "#fff3cd",
                "border": "1px solid #d39e00",
            }
        )

    for issue in issues:
        if not issue.get("record_uuid"):
            continue
        styles.append(
            {
                "if": {
                    "filter_query": f'{{__record_uuid}} = "{issue["record_uuid"]}"',
                    "column_id": issue["column_id"],
                },
                "backgroundColor": "#f8d7da",
                "border": "1px solid #dc3545",
            }
        )

    return styles


def build_reporting_table_columns(edit_mode=False):
    columns = []
    for col in REPORT_TABLE_COLUMNS:
        next_col = col.copy()
        if next_col["id"] in REPORT_EDITABLE_COLUMNS:
            next_col["editable"] = bool(edit_mode)
        columns.append(next_col)
    return columns


def compute_report_zones_from_df(df, max_hr_input=None):
    if df is None or df.empty:
        return ZONES_DEFAULT_ROWS

    zones_df = df.copy()
    for c in ["hr_bpm", "lactate_mmol", "actual_po_w", "rate_spm"]:
        if c not in zones_df.columns:
            zones_df[c] = None

    zones_df["hr_bpm"] = pd.to_numeric(zones_df["hr_bpm"], errors="coerce")
    zones_df["lactate_mmol"] = pd.to_numeric(zones_df["lactate_mmol"], errors="coerce")
    zones_df["actual_po_w"] = pd.to_numeric(zones_df["actual_po_w"], errors="coerce")
    zones_df["rate_spm"] = pd.to_numeric(zones_df["rate_spm"], errors="coerce")

    df_la_hr = zones_df.dropna(subset=["lactate_mmol", "hr_bpm"]).copy()
    if len(df_la_hr) < 2:
        return ZONES_DEFAULT_ROWS

    if max_hr_input is not None and max_hr_input != "":
        hr_max = float(max_hr_input)
    else:
        hr_max = zones_df["hr_bpm"].max()
        if pd.isna(hr_max):
            hr_max = df_la_hr["hr_bpm"].max()

    hr_max = float(hr_max)

    d_la_hr = df_la_hr.groupby("lactate_mmol", as_index=False)["hr_bpm"].mean().sort_values("lactate_mmol")
    la_vals = d_la_hr["lactate_mmol"].to_numpy(dtype=float)
    hr_vals = d_la_hr["hr_bpm"].to_numpy(dtype=float)

    def hr_at_la(target_la):
        if target_la <= la_vals.min():
            return float(hr_vals[0])
        if target_la >= la_vals.max():
            return float(hr_vals[-1])
        return float(np.interp(float(target_la), la_vals, hr_vals))

    df_la_po = zones_df.dropna(subset=["lactate_mmol", "actual_po_w"]).copy()
    po_at_la_ok = len(df_la_po) >= 2
    if po_at_la_ok:
        d_la_po = df_la_po.groupby("lactate_mmol", as_index=False)["actual_po_w"].mean().sort_values("lactate_mmol")
        la_po_vals = d_la_po["lactate_mmol"].to_numpy(dtype=float)
        po_vals = d_la_po["actual_po_w"].to_numpy(dtype=float)

        def po_at_la(target_la):
            if target_la <= la_po_vals.min():
                return float(po_vals[0])
            if target_la >= la_po_vals.max():
                return float(po_vals[-1])
            return float(np.interp(float(target_la), la_po_vals, po_vals))
    else:
        po_at_la = None  # noqa

    def hr_at_po(target_po):
        return _interp_y_at_x(zones_df, "actual_po_w", "hr_bpm", target_po)

    def po_at_hr(target_hr):
        return _interp_y_at_x(zones_df, "hr_bpm", "actual_po_w", target_hr)

    def rate_at_hr(target_hr):
        return _interp_y_at_x(zones_df, "hr_bpm", "rate_spm", target_hr)

    def split_from_po(target_po):
        if target_po is None:
            return None
        return format_split_mmss(estimate_split_seconds(target_po))

    def zone_row(zone_code, label, hr_low, hr_high, po_low, po_high, notes=""):
        rate_low = rate_at_hr(hr_low) if hr_low is not None else None
        rate_high = rate_at_hr(hr_high) if hr_high is not None else None

        return {
            "Zone": f"{zone_code}/{label}",
            "HR_low": round(hr_low, 0) if hr_low is not None else None,
            "HR_high": round(hr_high, 0) if hr_high is not None else None,
            "PO_low": round(po_low, 1) if po_low is not None else None,
            "PO_high": round(po_high, 1) if po_high is not None else None,
            "Split_low": split_from_po(po_low),
            "Split_high": split_from_po(po_high),
            "Rate_low": round(rate_low, 1) if rate_low is not None else None,
            "Rate_high": round(rate_high, 1) if rate_high is not None else None,
            "Notes": notes or "",
        }

    LA_LOW_C6 = 1.5
    LA_2 = 2.0
    LA_4 = 4.0

    hr_15 = hr_at_la(LA_LOW_C6)
    hr_2 = hr_at_la(LA_2)
    hr_4 = hr_at_la(LA_4)

    hr_15, hr_2, hr_4 = sorted([hr_15, hr_2, hr_4])

    z1_lo = 100.0
    z1_hi = hr_15
    z2_lo = hr_15
    z2_hi = hr_2

    if not po_at_la_ok:
        return [
            zone_row("Z1", "C7", z1_lo, z1_hi, po_at_hr(z1_lo), po_at_hr(z1_hi), notes="HR-based"),
            zone_row("Z2", "C6", z2_lo, z2_hi, po_at_hr(z2_lo), po_at_hr(z2_hi), notes="HR-based"),
            zone_row("Z3", "C5", z2_hi, hr_4, po_at_hr(z2_hi), po_at_hr(hr_4), notes="Fallback (no La→PO)"),
            zone_row("Z4", "C4", None, None, None, None, notes="Fallback (no La→PO)"),
            zone_row("Z5", "C3", hr_4, (hr_4 + hr_max) / 2.0, po_at_hr(hr_4), None, notes="HR-based"),
            zone_row("Z6", "C2/C1", (hr_4 + hr_max) / 2.0, hr_max, None, None, notes="HR-based"),
        ]

    po_2 = float(po_at_la(LA_2))
    po_4 = float(po_at_la(LA_4))
    po_lo, po_hi = (po_2, po_4) if po_2 <= po_4 else (po_4, po_2)
    po_mid = (po_lo + po_hi) / 2.0

    hr_at_po2 = hr_at_po(po_lo)
    hr_at_pomid = hr_at_po(po_mid)
    hr_at_po4 = hr_at_po(po_hi)

    if hr_at_po2 is None:
        hr_at_po2 = hr_2
    if hr_at_pomid is None:
        hr_at_pomid = (hr_2 + hr_4) / 2.0
    if hr_at_po4 is None:
        hr_at_po4 = hr_4

    z3_po_lo, z3_po_hi = po_lo, po_mid
    z3_hr_lo, z3_hr_hi = float(hr_at_po2), float(hr_at_pomid)

    z4_po_lo, z4_po_hi = po_mid, po_hi
    z4_hr_lo, z4_hr_hi = float(hr_at_pomid), float(hr_at_po4)

    z5_hr_lo = hr_4
    z5_hr_hi = (hr_4 + hr_max) / 2.0
    z6_hr_lo = z5_hr_hi
    z6_hr_hi = hr_max

    z5_po_low = po_at_hr(z5_hr_lo)
    z6_po_low = po_at_hr(z6_hr_lo)

    return [
        zone_row("Z1", "C7", z1_lo, z1_hi, po_at_hr(z1_lo), po_at_hr(z1_hi), notes="100 bpm -> low C6 (HR@1.5)"),
        zone_row("Z2", "C6", z2_lo, z2_hi, po_at_hr(z2_lo), po_at_hr(z2_hi), notes="1.5-2 mmol HR"),
        zone_row("Z3", "C5", z3_hr_lo, z3_hr_hi, z3_po_lo, z3_po_hi, notes="2 mmol W -> midpoint (2-4 mmol W)"),
        zone_row("Z4", "C4", z4_hr_lo, z4_hr_hi, z4_po_lo, z4_po_hi, notes="Midpoint -> 4 mmol W"),
        zone_row("Z5", "C3", z5_hr_lo, z5_hr_hi, z5_po_low, None, notes="4 mmol HR -> halfway to max HR"),
        zone_row("Z6", "C2/C1", z6_hr_lo, z6_hr_hi, z6_po_low, None, notes="Halfway -> max HR"),
    ]


def dataframe_to_store_records(df):
    if df is None or df.empty:
        return []

    out = df.copy()
    out = out.replace({np.nan: None})

    for col in ["test_date", "session_ts"]:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: x.isoformat() if hasattr(x, "isoformat") and pd.notna(x) else None
            )

    return out.to_dict("records")


def original_records_by_uuid(records):
    rows = {}
    for rec in records or []:
        payload = _extract_record_payload(rec)
        record_uuid = payload.get("__record_uuid") or rec.get("uuid") if isinstance(rec, dict) else None
        if record_uuid:
            rows[str(record_uuid)] = payload
    return rows


def merge_replacement_record(updated_row, replacement_record):
    replacement_payload = _extract_record_payload(replacement_record)
    merged = updated_row.copy()

    if replacement_payload.get("__record_uuid"):
        merged["__record_uuid"] = replacement_payload["__record_uuid"]
    if replacement_payload.get("__dataset_uuid"):
        merged["__dataset_uuid"] = replacement_payload["__dataset_uuid"]

    return merged


def fetch_single_record_from_dataset(dataset_uuid):
    if not dataset_uuid:
        return None

    records = wc.list_records(
        source_uuid=VO2_STEP_SOURCE_UUID,
        role="primary",
        page_size=10,
        extra_params={"dataset_uuid": dataset_uuid},
    )
    return records[0] if records else None

def _extract_record_payload(rec):
    """
    Try to pull the actual ingested payload out of a warehouse record.
    """
    if not isinstance(rec, dict):
        return {}

    # Raw fields already top-level
    if "profile_id" in rec or "session_id" in rec or "test_date" in rec:
        return rec

    # Common wrapped shapes
    for key in ["data", "record", "raw"]:
        if isinstance(rec.get(key), dict):
            payload = rec[key].copy()
            payload["__record_uuid"] = rec.get("uuid")
            payload["__dataset_uuid"] = rec.get("dataset_uuid")
            return payload
            payload = rec[key].copy()
            payload["__record_uuid"] = rec.get("uuid")
            payload["__dataset_uuid"] = rec.get("dataset_uuid")
            return payload

    return rec


def normalize_records_to_df(records):
    expected_cols = ["__record_uuid", "__dataset_uuid"] + REPORT_DATA_COLUMNS

    if not records:
        return pd.DataFrame(columns=expected_cols)

    extracted = [_extract_record_payload(r) for r in records]
    df = pd.DataFrame(extracted)

    for c in expected_cols:
        if c not in df.columns:
            df[c] = None

    for c in REPORT_NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce")
    df["session_ts"] = pd.to_datetime(df["session_ts"], errors="coerce")

    return df


def apply_local_filters(df, profile_id=None, start_date=None, end_date=None, test_type=None, mode=None):
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "profile_id", "session_id", "session_ts", "test_date", "body_mass_kg",
            "test_type", "mode", "notes", "step_no", "step_type", "target_po_w",
            "actual_po_w", "hr_bpm", "lactate_mmol", "vo2", "rate_spm",
            "split_sec_per_500", "rpe", "time_s"
        ])

    dff = df.copy()

    if profile_id not in (None, "", []):
        dff = dff[dff["profile_id"] == pd.to_numeric(profile_id, errors="coerce")]

    if start_date:
        dff = dff[dff["test_date"] >= pd.to_datetime(start_date)]

    if end_date:
        # inclusive end date
        dff = dff[dff["test_date"] < (pd.to_datetime(end_date) + pd.Timedelta(days=1))]

    if test_type not in (None, "", "all"):
        dff = dff[dff["test_type"] == test_type]

    if mode not in (None, "", "all"):
        dff = dff[dff["mode"] == mode]

    dff = dff.sort_values(
        by=["test_date", "session_ts", "session_id", "step_no"],
        ascending=[False, False, False, True],
        na_position="last",
    )

    return dff


def fetch_step_test_data_from_warehouse(
    source_uuid,
    profile_id=None,
    start_date=None,
    end_date=None,
    test_type=None,
    mode=None,
):
    """
    Pull records from warehouse, then filter locally by ingested test_date.
    """
    if not source_uuid:
        raise ValueError("VO2_STEP_SOURCE_UUID is not set.")

    records = wc.list_records(
        source_uuid=source_uuid,
        subject=int(profile_id) if profile_id not in (None, "", []) else None,
        role="primary",
        page_size=500,
    )

    df = normalize_records_to_df(records)

    return apply_local_filters(
        df,
        profile_id=profile_id,
        start_date=start_date,
        end_date=end_date,
        test_type=test_type,
        mode=mode,
    )
def add_athlete_names(df, athlete_options):
    if df is None or df.empty:
        df = pd.DataFrame(columns=["profile_id"])
    dff = df.copy()

    id_to_name = {}
    for opt in athlete_options or []:
        try:
            id_to_name[int(opt["value"])] = opt["label"]
        except Exception:
            continue

    dff["athlete_name"] = dff["profile_id"].map(id_to_name).fillna(
        dff["profile_id"].astype("Int64").astype(str)
    )
    return dff


# =========================================================
# LAYOUT
# =========================================================
layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.Div("PHYSIOLOGY", className="page-eyebrow"),
                    html.H1("Reporting", className="mb-1"),
                    html.P(
                        "Pull step test data from the warehouse and review athlete trends.",
                        className="text-muted mb-0",
                    ),
                ]
            ),
            className="page-title-header align-items-center mt-4 mb-4",
        ),

        dcc.Store(id="reporting-athlete-options-store"),
        dcc.Store(id="reporting-data-store"),
        dcc.Store(id="reporting-edit-mode-store", data=False),

        dbc.Tabs(
            [
                dbc.Tab(
                    label="Step Tests",
                    tab_id="tab-report-step-test",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    make_card(
                                        "Filters",
                                        [
                                            dbc.Label("Athlete"),
                                            dcc.Dropdown(
                                                id="reporting-athlete",
                                                options=[],
                                                placeholder="All athletes",
                                                value=None,
                                                clearable=True,
                                            ),
                                            html.Br(),

                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Start Date"),
                                                            dcc.DatePickerSingle(
                                                                id="reporting-start-date",
                                                                date=(date.today() - timedelta(days=365)).isoformat(),
                                                                display_format="YYYY-MM-DD",
                                                                clearable=True,
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("End Date"),
                                                            dcc.DatePickerSingle(
                                                                id="reporting-end-date",
                                                                date=(date.today() + timedelta(days=365)).isoformat(),
                                                                display_format="YYYY-MM-DD",
                                                                clearable=True,
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ],
                                                className="g-2",
                                            ),
                                            html.Br(),

                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Test Type"),
                                                            dcc.Dropdown(
                                                                id="reporting-test-type",
                                                                options=[
                                                                    {"label": "All", "value": "all"},
                                                                    {"label": "Erg C2", "value": "erg_C2"},
                                                                    {"label": "Erg RP3", "value": "erg_RP3"},
                                                                    {"label": "On-Water", "value": "row"},
                                                                    {"label": "Bike", "value": "bike"},
                                                                    {"label": "Other", "value": "other"},
                                                                ],
                                                                value="all",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Mode"),
                                                            dcc.Dropdown(
                                                                id="reporting-mode",
                                                                options=[
                                                                    {"label": "All", "value": "all"},
                                                                    {"label": "Max", "value": "Max"},
                                                                    {"label": "Submax", "value": "Submax"},
                                                                ],
                                                                value="all",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ],
                                                className="g-2",
                                            ),
                                            html.Br(),
                                            dbc.Label("Max HR (bpm)"),
                                            dbc.Input(
                                                id="reporting-max-hr",
                                                type="number",
                                                min=100,
                                                max=240,
                                                step=1,
                                                placeholder="optional",
                                                value=None,
                                            ),
                                            html.Br(),

                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dbc.Button(
                                                            "Load Data",
                                                            id="reporting-load-btn",
                                                            color="primary",
                                                            className="w-100",
                                                        ),
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Button(
                                                            "Download CSV",
                                                            id="reporting-download-btn",
                                                            color="info",
                                                            outline=True,
                                                            className="w-100",
                                                        ),
                                                        md=6,
                                                    ),
                                                ],
                                                className="g-2",
                                            ),
                                            html.Br(),
                                            dbc.Button(
                                                "Edit Data",
                                                id="reporting-edit-btn",
                                                color="secondary",
                                                outline=True,
                                                className="w-100",
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "Revert Changes",
                                                        id="reporting-revert-btn",
                                                        color="secondary",
                                                        outline=True,
                                                        className="w-100",
                                                        disabled=True,
                                                    ),
                                                    dbc.Button(
                                                        "Review Changes",
                                                        id="reporting-update-btn",
                                                        color="success",
                                                        className="w-100 mt-2",
                                                        disabled=True,
                                                    ),
                                                ],
                                                className="mt-2",
                                            ),
                                            dcc.Download(id="reporting-download"),
                                            html.Hr(),
                                            dbc.Alert(id="reporting-status-msg", is_open=False),
                                            dbc.Alert(id="reporting-update-msg", is_open=False),
                                        ],
                                    ),
                                    md=3,
                                ),

                                dbc.Col(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(make_card("Rows", html.H4(id="reporting-rows", className="m-0")), md=3),
                                                dbc.Col(make_card("Sessions", html.H4(id="reporting-sessions", className="m-0")), md=3),
                                                dbc.Col(make_card("Avg PO", html.H4(id="reporting-avg-po", className="m-0")), md=3),
                                                dbc.Col(make_card("Avg HR", html.H4(id="reporting-avg-hr", className="m-0")), md=3),
                                            ],
                                            className="g-2 mb-3",
                                        ),

                                        make_card(
                                            "Data Table",
                                            dash_table.DataTable(
                                                id="reporting-table",
                                                data=[],
                                                columns=build_reporting_table_columns(False),
                                                hidden_columns=["__record_uuid", "profile_id", "__row_issues"],
                                                editable=True,
                                                page_action="native",
                                                page_size=15,
                                                sort_action="native",
                                                filter_action="native",
                                                dropdown={
                                                    col: {"options": options}
                                                    for col, options in REPORT_DROPDOWN_OPTIONS.items()
                                                },
                                                style_table={"overflowX": "auto"},
                                                style_cell={
                                                    "padding": "8px",
                                                    "fontFamily": "system-ui",
                                                    "fontSize": 14,
                                                    "textAlign": "left",
                                                    "minWidth": "100px",
                                                    "maxWidth": "220px",
                                                    "whiteSpace": "normal",
                                                },
                                                style_header={"fontWeight": "600"},
                                                style_header_conditional=[
                                                    {
                                                        "if": {"column_id": col},
                                                        "backgroundColor": "#e8f4ff",
                                                        "color": "#0b4f79",
                                                    }
                                                    for col in REPORT_EDITABLE_COLUMNS
                                                ],
                                                style_data_conditional=[
                                                    {
                                                        "if": {"column_id": col},
                                                        "backgroundColor": "#f3f9ff",
                                                    }
                                                    for col in REPORT_EDITABLE_COLUMNS
                                                ],
                                            ),
                                        ),
                                        html.Br(),
                                        make_card(
                                            "HR Training Zones (from Lactate Thresholds)",
                                            dash_table.DataTable(
                                                id="reporting-zones-table",
                                                data=ZONES_DEFAULT_ROWS,
                                                columns=ZONES_COLUMNS,
                                                editable=False,
                                                page_action="none",
                                                style_table={"overflowX": "auto"},
                                                style_cell={
                                                    "padding": "8px",
                                                    "fontFamily": "system-ui",
                                                    "fontSize": 14,
                                                    "textAlign": "left",
                                                    "minWidth": "90px",
                                                    "maxWidth": "220px",
                                                    "whiteSpace": "normal",
                                                },
                                                style_header={"fontWeight": "600"},
                                            ),
                                        ),
                                    ],
                                    md=9,
                                ),
                            ],
                            className="g-3",
                        ),

                        html.Hr(),

                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="reporting-po-hr-plot", config={"displayModeBar": False}), md=6),
                                dbc.Col(dcc.Graph(id="reporting-po-la-plot", config={"displayModeBar": False}), md=6),
                            ],
                            className="g-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="reporting-session-trend-plot", config={"displayModeBar": False}), md=12),
                            ],
                            className="g-2 mt-2",
                        ),

                        dbc.Modal(
                            [
                                dbc.ModalHeader(dbc.ModalTitle("Review Warehouse Changes")),
                                dbc.ModalBody(id="reporting-change-summary"),
                                dbc.ModalFooter(
                                    [
                                        dbc.Button("Cancel", id="reporting-cancel-update-btn", color="secondary", outline=True),
                                        dbc.Button(
                                            "Update Warehouse",
                                            id="reporting-confirm-update-btn",
                                            color="success",
                                            disabled=True,
                                        ),
                                    ]
                                ),
                            ],
                            id="reporting-review-modal",
                            size="lg",
                            is_open=False,
                            scrollable=True,
                        ),
                    ],
                ),

                # The erg tab carries its own filters, table and review
                # modal; see erg_editor.py.
                dbc.Tab(
                    label="Erg Scores",
                    tab_id="tab-report-erg",
                    children=[erg_editor.layout],
                ),
            ],
            id="reporting-tabs",
            active_tab="tab-report-step-test",
            className="mt-2",
        ),
    ],
    fluid=True,
)


# =========================================================
# LOAD ATHLETE OPTIONS
# =========================================================
@dash.callback(
    Output("reporting-athlete-options-store", "data"),
    Input("reporting-load-btn", "id"),
)
def load_athlete_options(_):
    try:
        token = auth.get_token()
        filters = {"sport_org_id": 13}
        names = fetch_profiles(token, filters)
    except Exception:
        raise PreventUpdate

    return [
        {
            "label": f"{p['person']['first_name']} {p['person']['last_name']}",
            "value": int(p["id"]),
        }
        for p in names
    ]


@dash.callback(
    Output("reporting-athlete", "options"),
    Input("reporting-athlete-options-store", "data"),
)
def apply_athlete_options(options):
    return options or []


# =========================================================
# LOAD DATA
# =========================================================
@dash.callback(
    Output("reporting-data-store", "data"),
    Output("reporting-status-msg", "children"),
    Output("reporting-status-msg", "color"),
    Output("reporting-status-msg", "is_open"),
    Input("reporting-load-btn", "n_clicks"),
    State("reporting-athlete", "value"),
    State("reporting-start-date", "date"),
    State("reporting-end-date", "date"),
    State("reporting-test-type", "value"),
    State("reporting-mode", "value"),
    prevent_initial_call=True,
)
def load_reporting_data(n_clicks, athlete_id, start_date, end_date, test_type, mode):
    if not n_clicks:
        raise PreventUpdate

    try:
        df = fetch_step_test_data_from_warehouse(
            source_uuid=VO2_STEP_SOURCE_UUID,
            profile_id=athlete_id,
            start_date=safe_date_str(start_date),
            end_date=safe_date_str(end_date),
            test_type=test_type,
            mode=mode,
        )

        if df.empty:
            return [], "No rows found for the selected filters.", "warning", True

        records = dataframe_to_store_records(df)
        return records, f"Loaded {len(df)} row(s) from warehouse.", "success", True

    except (WarehouseClientError, ValueError, AttributeError) as e:
        return [], f"Load failed: {e}", "danger", True

    except Exception as e:
        return [], f"Unexpected error: {e}", "danger", True


# =========================================================
# TABLE + SUMMARY
# =========================================================
@dash.callback(
    Output("reporting-table", "data"),
    Output("reporting-rows", "children"),
    Output("reporting-sessions", "children"),
    Output("reporting-avg-po", "children"),
    Output("reporting-avg-hr", "children"),
    Input("reporting-data-store", "data"),
    State("reporting-athlete-options-store", "data"),
)
def update_reporting_table(records, athlete_options):
    df = normalize_records_to_df(records)

    if df.empty:
        return [], "0", "0", "—", "—"

    df_display = df.copy()
    df_display = add_athlete_names(df_display, athlete_options)
    df_display["test_date"] = df_display["test_date"].astype(str)
    df_display["split_sec_per_500"] = df_display["split_sec_per_500"].apply(format_split_mmss)
    df_display["__row_issues"] = ""

    rows_txt = str(len(df_display))
    sessions_txt = str(df_display["session_id"].dropna().nunique())

    avg_po = df["actual_po_w"].mean(skipna=True)
    avg_hr = df["hr_bpm"].mean(skipna=True)

    avg_po_txt = f"{avg_po:.1f} W" if pd.notna(avg_po) else "—"
    avg_hr_txt = f"{avg_hr:.1f} bpm" if pd.notna(avg_hr) else "—"

    return df_display.to_dict("records"), rows_txt, sessions_txt, avg_po_txt, avg_hr_txt


@dash.callback(
    Output("reporting-zones-table", "data"),
    Input("reporting-table", "data"),
    Input("reporting-max-hr", "value"),
)
def update_reporting_zones(table_rows, max_hr_input):
    df = normalize_records_to_df(table_rows)
    return compute_report_zones_from_df(df, max_hr_input)


@dash.callback(
    Output("reporting-edit-mode-store", "data"),
    Input("reporting-edit-btn", "n_clicks"),
    Input("reporting-data-store", "data"),
    State("reporting-edit-mode-store", "data"),
)
def toggle_reporting_edit_mode(edit_clicks, records, edit_mode):
    trigger = ctx.triggered_id
    if trigger == "reporting-edit-btn":
        return not bool(edit_mode)
    if trigger == "reporting-data-store":
        return False
    return bool(edit_mode)


@dash.callback(
    Output("reporting-table", "columns"),
    Output("reporting-table", "style_data_conditional"),
    Output("reporting-edit-btn", "children"),
    Output("reporting-edit-btn", "color"),
    Output("reporting-revert-btn", "disabled"),
    Output("reporting-update-btn", "disabled"),
    Input("reporting-edit-mode-store", "data"),
    Input("reporting-table", "data"),
    State("reporting-data-store", "data"),
)
def update_reporting_edit_controls(edit_mode, table_rows, original_records):
    changes = get_changed_cells(original_records, table_rows)
    issues = validate_reporting_rows(table_rows)
    styles = build_reporting_table_styles(changes, issues)
    has_changes = bool(changes)
    edit_mode = bool(edit_mode)

    return (
        build_reporting_table_columns(edit_mode),
        styles,
        "Exit Edit Mode" if edit_mode else "Edit Data",
        "warning" if edit_mode else "secondary",
        not (edit_mode and has_changes),
        not (edit_mode and has_changes),
    )


@dash.callback(
    Output("reporting-table", "data", allow_duplicate=True),
    Input("reporting-revert-btn", "n_clicks"),
    State("reporting-data-store", "data"),
    State("reporting-athlete-options-store", "data"),
    prevent_initial_call=True,
)
def revert_reporting_table_changes(n_clicks, records, athlete_options):
    if not n_clicks:
        raise PreventUpdate

    table_rows, _, _, _, _ = update_reporting_table(records, athlete_options)
    return table_rows


@dash.callback(
    Output("reporting-review-modal", "is_open"),
    Output("reporting-change-summary", "children"),
    Output("reporting-confirm-update-btn", "disabled"),
    Input("reporting-update-btn", "n_clicks"),
    Input("reporting-cancel-update-btn", "n_clicks"),
    State("reporting-review-modal", "is_open"),
    State("reporting-table", "data"),
    State("reporting-data-store", "data"),
    prevent_initial_call=True,
)
def toggle_reporting_review_modal(review_clicks, cancel_clicks, is_open, table_rows, original_records):
    trigger = ctx.triggered_id
    if trigger == "reporting-cancel-update-btn":
        return False, no_update, True

    if trigger != "reporting-update-btn":
        raise PreventUpdate

    changes = get_changed_cells(original_records, table_rows)
    issues = validate_reporting_rows(table_rows)
    return True, build_change_summary(changes, issues), bool(issues or not changes)


@dash.callback(
    Output("reporting-data-store", "data", allow_duplicate=True),
    Output("reporting-update-msg", "children"),
    Output("reporting-update-msg", "color"),
    Output("reporting-update-msg", "is_open"),
    Output("reporting-review-modal", "is_open", allow_duplicate=True),
    Input("reporting-confirm-update-btn", "n_clicks"),
    State("reporting-table", "data"),
    State("reporting-data-store", "data"),
    prevent_initial_call=True,
)
def update_warehouse_records(n_clicks, edited_rows, original_records):
    if not n_clicks:
        raise PreventUpdate

    original_df = normalize_records_to_df(original_records)
    edited_df = pd.DataFrame(edited_rows or [])

    if original_df.empty or edited_df.empty:
        return no_update, "No rows loaded to update.", "warning", True, False

    if "__record_uuid" not in original_df.columns or "__record_uuid" not in edited_df.columns:
        return no_update, "Update failed: warehouse record UUIDs are missing. Reload the data and try again.", "danger", True, False

    issues = validate_reporting_rows(edited_rows)
    if issues:
        return no_update, "Update blocked: resolve validation issues before saving.", "danger", True, True

    original_by_uuid = {
        str(row["__record_uuid"]): row
        for row in original_df.to_dict("records")
        if row.get("__record_uuid")
    }
    original_payload_by_uuid = original_records_by_uuid(original_records)

    updated_count = 0
    local_updates = {}

    try:
        for edited_row in edited_df.to_dict("records"):
            record_uuid = edited_row.get("__record_uuid")
            if not record_uuid:
                continue

            record_uuid = str(record_uuid)
            original_row = original_by_uuid.get(record_uuid)
            if not original_row:
                continue

            changed_cols = [
                col
                for col in REPORT_EDITABLE_COLUMNS
                if not values_equal(edited_row.get(col), original_row.get(col), col)
            ]

            if not changed_cols:
                continue

            updated_row = original_payload_by_uuid.get(record_uuid, original_row).copy()
            for col in changed_cols:
                updated_row[col] = clean_cell_value(edited_row.get(col), col)
            updated_row["__record_uuid"] = record_uuid

            patch_payload = {"data": row_to_warehouse_payload(updated_row)}
            if updated_row.get("__dataset_uuid"):
                patch_payload["dataset"] = updated_row["__dataset_uuid"]

            try:
                wc.patch_record(
                    record_uuid=record_uuid,
                    data=patch_payload,
                )
                replacement_uuid = record_uuid
            except WarehouseClientError as patch_error:
                try:
                    wc.put_record(
                        record_uuid=record_uuid,
                        data=patch_payload,
                    )
                    replacement_uuid = record_uuid
                except WarehouseClientError as put_error:
                    try:
                        if not patch_payload.get("dataset"):
                            raise WarehouseClientError("Skipped direct create: missing dataset UUID.")
                        replacement_record = wc.create_record(data=patch_payload)
                    except WarehouseClientError as create_error:
                        try:
                            replacement_dataset, created_count = wc.ingest_raw(
                                source_uuid=VO2_STEP_SOURCE_UUID,
                                records=[patch_payload["data"]],
                                subject_field="profile_id",
                                validate_client_side=False,
                            )
                            if created_count != 1:
                                raise WarehouseClientError(
                                    f"Ingestion created {created_count} records instead of 1."
                                )
                            replacement_record = fetch_single_record_from_dataset(
                                replacement_dataset.get("uuid")
                            )
                            if not replacement_record:
                                raise WarehouseClientError(
                                    "Ingestion succeeded but the replacement record could not be reloaded."
                                )
                        except WarehouseClientError as ingest_error:
                            raise WarehouseClientError(
                                f"PATCH, PUT, direct create, and ingestion replacement all failed for {record_uuid}. "
                                f"PATCH: {patch_error}. PUT: {put_error}. "
                                f"CREATE: {create_error}. INGEST: {ingest_error}."
                            ) from ingest_error

                    replacement_uuid = replacement_record.get("uuid")
                    try:
                        wc.delete_record(record_uuid=record_uuid)
                    except WarehouseClientError as delete_error:
                        raise WarehouseClientError(
                            "Replacement record was created, but deleting the old record failed. "
                            f"Old record: {record_uuid}. Replacement record: {replacement_uuid}. "
                            f"DELETE: {delete_error}."
                        ) from delete_error

                    updated_row = merge_replacement_record(updated_row, replacement_record)

            local_updates[record_uuid] = updated_row
            if replacement_uuid and replacement_uuid != record_uuid:
                local_updates[str(replacement_uuid)] = updated_row
            updated_count += 1

    except WarehouseClientError as e:
        return no_update, f"Update failed: {e}", "danger", True, False
    except Exception as e:
        return no_update, f"Unexpected update error: {e}", "danger", True, False

    if not updated_count:
        return no_update, "No editable changes detected.", "info", True, False

    refreshed = []
    for row in original_df.to_dict("records"):
        record_uuid = str(row.get("__record_uuid")) if row.get("__record_uuid") else None
        refreshed.append(local_updates.get(record_uuid, row))

    return (
        dataframe_to_store_records(pd.DataFrame(refreshed)),
        f"Updated {updated_count} warehouse record(s).",
        "success",
        True,
        False,
    )


# =========================================================
# PLOTS
# =========================================================
@dash.callback(
    Output("reporting-po-hr-plot", "figure"),
    Output("reporting-po-la-plot", "figure"),
    Output("reporting-session-trend-plot", "figure"),
    Input("reporting-data-store", "data"),
    State("reporting-athlete-options-store", "data"),
)
def update_reporting_plots(records, athlete_options):
    df = normalize_records_to_df(records)

    # ---------- Empty figs ----------
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        title="No data",
    )

    if df.empty:
        return empty_fig, empty_fig, empty_fig

    df = add_athlete_names(df, athlete_options)
    df = make_session_label(df)

    # ---------- Empty figs ----------
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        title="No data",
    )

    if df.empty:
        return empty_fig, empty_fig, empty_fig

    # sort for cleaner line drawing
    df = df.sort_values(["athlete_name", "test_date", "session_id", "step_no", "actual_po_w"])

    # =========================================================
    # PO vs HR with linear fits per session
    # =========================================================
    df_hr = df.dropna(subset=["actual_po_w", "hr_bpm"]).copy()

    fig_hr = go.Figure()
    if df_hr.empty:
        fig_hr = empty_fig
    else:
        # color by session/test day
        session_labels_hr = list(df_hr["session_label"].dropna().unique())
        palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24

        color_map_hr = {
            lab: palette[i % len(palette)]
            for i, lab in enumerate(session_labels_hr)
        }

        for sess_key, g in df_hr.groupby("session_key", dropna=False):
            sess_label = g["session_label"].iloc[0]
            session_keys_hr = list(df_hr["session_key"].dropna().unique())
            color_map_hr = {
                key: palette[i % len(palette)]
                for i, key in enumerate(session_keys_hr)
            }
            g = g.sort_values("actual_po_w")
            color = color_map_hr.get(sess_key, None)

            athlete_name = g["athlete_name"].iloc[0] if "athlete_name" in g.columns and len(g) else "Athlete"

            # scatter points
            fig_hr.add_trace(
                go.Scatter(
                    x=g["actual_po_w"],
                    y=g["hr_bpm"],
                    mode="markers",
                    name=f"{sess_label} points",
                    legendgroup=str(sess_label),
                    marker=dict(size=8, color=color),
                    customdata=np.stack(
                        [
                            g["athlete_name"].astype(str),
                            g["test_date"].astype(str),
                            g["step_no"].astype(str),
                            g["test_type"].astype(str),
                            g["mode"].astype(str),
                            g["session_id"].astype(str),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Session: %{customdata[5]}<br>"
                        "Date: %{customdata[1]}<br>"
                        "Step: %{customdata[2]}<br>"
                        "Test Type: %{customdata[3]}<br>"
                        "Mode: %{customdata[4]}<br>"
                        "PO: %{x:.1f} W<br>"
                        "HR: %{y:.1f} bpm<extra></extra>"
                    ),
                )
            )

            # linear fit if enough data
            fit_df = g.dropna(subset=["actual_po_w", "hr_bpm"]).copy()
            fit_df = fit_df.groupby("actual_po_w", as_index=False)["hr_bpm"].mean().sort_values("actual_po_w")

            if len(fit_df) >= 2 and fit_df["actual_po_w"].nunique() >= 2:
                x = fit_df["actual_po_w"].to_numpy(dtype=float)
                y = fit_df["hr_bpm"].to_numpy(dtype=float)

                try:
                    coeffs = np.polyfit(x, y, deg=1)
                    poly = np.poly1d(coeffs)

                    x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                    y_fit = poly(x_fit)

                    # optional R²
                    y_pred = poly(x)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

                    fig_hr.add_trace(
                        go.Scatter(
                            x=x_fit,
                            y=y_fit,
                            mode="lines",
                            name=f"{sess_label} fit",
                            legendgroup=str(sess_label),
                            line=dict(width=3, color=color),
                            hovertemplate=(
                                f"{athlete_name}<br>"
                                f"{sess_label}<br>"
                                f"Linear fit"
                                + (f"<br>R²: {r2:.3f}" if pd.notna(r2) else "")
                                + "<br>PO: %{x:.1f} W<br>HR: %{y:.1f} bpm<extra></extra>"
                            ),
                        )
                    )
                except Exception:
                    pass

        fig_hr.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            title="Heart Rate vs Actual PO (Linear Fit by Test Day)",
            xaxis_title="Actual PO (W)",
            yaxis_title="Heart Rate (bpm)",
            legend_title_text="Test Day / Session",
        )

    # =========================================================
    # PO vs Lactate with quadratic fits per session
    # =========================================================
    df_la = df.dropna(subset=["actual_po_w", "lactate_mmol"]).copy()

    fig_la = go.Figure()
    if df_la.empty:
        fig_la = empty_fig
    else:
        session_labels_la = list(df_la["session_label"].dropna().unique())
        palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24

        color_map_la = {
            lab: palette[i % len(palette)]
            for i, lab in enumerate(session_labels_la)
        }

        for sess_key, g in df_la.groupby("session_key", dropna=False):
            sess_label = g["session_label"].iloc[0]
            session_keys_la = list(df_la["session_key"].dropna().unique())
            color_map_la = {
                key: palette[i % len(palette)]
                for i, key in enumerate(session_keys_la)
            }
            color = color_map_la.get(sess_key, None)

            athlete_name = g["athlete_name"].iloc[0] if "athlete_name" in g.columns and len(g) else "Athlete"

            # scatter points
            fig_la.add_trace(
                go.Scatter(
                    x=g["actual_po_w"],
                    y=g["lactate_mmol"],
                    mode="markers",
                    name=f"{sess_label} points",
                    legendgroup=str(sess_label),
                    marker=dict(size=8, color=color),
                    customdata=np.stack(
                        [
                            g["athlete_name"].astype(str),
                            g["test_date"].astype(str),
                            g["step_no"].astype(str),
                            g["test_type"].astype(str),
                            g["mode"].astype(str),
                            g["session_id"].astype(str),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Session: %{customdata[5]}<br>"
                        "Date: %{customdata[1]}<br>"
                        "Step: %{customdata[2]}<br>"
                        "Test Type: %{customdata[3]}<br>"
                        "Mode: %{customdata[4]}<br>"
                        "PO: %{x:.1f} W<br>"
                        "Lactate: %{y:.2f} mmol/L<extra></extra>"
                    ),
                )
            )

            # quadratic fit if enough points, otherwise linear fallback
            fit_df = g.dropna(subset=["actual_po_w", "lactate_mmol"]).copy()
            fit_df = (
                fit_df.groupby("actual_po_w", as_index=False)["lactate_mmol"]
                .mean()
                .sort_values("actual_po_w")
            )

            n_unique = fit_df["actual_po_w"].nunique()

            if n_unique >= 2:
                x = fit_df["actual_po_w"].to_numpy(dtype=float)
                y = fit_df["lactate_mmol"].to_numpy(dtype=float)

                deg = 2 if n_unique >= 3 else 1

                try:
                    coeffs = np.polyfit(x, y, deg=deg)
                    poly = np.poly1d(coeffs)

                    x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 250)
                    y_fit = poly(x_fit)

                    # optional R²
                    y_pred = poly(x)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

                    fit_name = "Quadratic fit" if deg == 2 else "Linear fit"

                    fig_la.add_trace(
                        go.Scatter(
                            x=x_fit,
                            y=y_fit,
                            mode="lines",
                            name=f"{sess_label} fit",
                            legendgroup=str(sess_label),
                            line=dict(width=3, color=color),
                            hovertemplate=(
                                f"{athlete_name}<br>"
                                f"{sess_label}<br>"
                                f"{fit_name}"
                                + (f"<br>R²: {r2:.3f}" if pd.notna(r2) else "")
                                + "<br>PO: %{x:.1f} W<br>Lactate: %{y:.2f} mmol/L<extra></extra>"
                            ),
                        )
                    )
                except Exception:
                    pass

        fig_la.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            title="Blood Lactate vs Actual PO (Fit by Test Day)",
            xaxis_title="Actual PO (W)",
            yaxis_title="Blood Lactate (mmol/L)",
            legend_title_text="Test Day / Session",
        )

    # =========================================================
    # Threshold trend
    # =========================================================
    trend_df = build_lactate_threshold_trend_df(df)

    if trend_df.empty:
        fig_trend = empty_fig
    else:
        trend_long = trend_df.melt(
            id_vars=[
                "session_id",
                "profile_id",
                "athlete_name",
                "test_date_dt",
                "fit_degree",
                "n_points",
                "max_po",
                "max_la",
            ],
            value_vars=["lt_2_w", "lt_4_w", "lt_6_w"],
            var_name="threshold",
            value_name="power_w",
        )

        threshold_map = {
            "lt_2_w": "2 mmol",
            "lt_4_w": "4 mmol",
            "lt_6_w": "6 mmol",
        }
        trend_long["threshold_label"] = trend_long["threshold"].map(threshold_map)
        trend_long = trend_long.dropna(subset=["test_date_dt", "power_w"]).copy()

        if trend_long.empty:
            fig_trend = empty_fig
        else:
            fig_trend = px.line(
                trend_long,
                x="test_date_dt",
                y="power_w",
                color="athlete_name",
                line_dash="threshold_label",
                markers=True,
                hover_data={
                    "session_id": True,
                    "athlete_name": True,
                    "threshold_label": True,
                    "power_w": ":.1f",
                    "fit_degree": True,
                    "n_points": True,
                    "max_po": ":.1f",
                    "max_la": ":.2f",
                    "test_date_dt": False,
                },
                title="Estimated Power at 2, 4, and 6 mmol Lactate Thresholds",
                labels={
                    "test_date_dt": "Test Date",
                    "power_w": "Estimated Power (W)",
                    "athlete_name": "Athlete",
                    "threshold_label": "Threshold",
                },
            )

            fig_trend.update_layout(
                template="plotly_white",
                margin=dict(l=20, r=20, t=50, b=20),
                legend_title_text="Athlete / Threshold",
            )

            fig_trend.update_traces(
                mode="lines+markers",
                marker=dict(size=8),
            )

    return fig_hr, fig_la, fig_trend
# =========================================================
# DOWNLOAD
# =========================================================
@dash.callback(
    Output("reporting-download", "data"),
    Input("reporting-download-btn", "n_clicks"),
    State("reporting-data-store", "data"),
    prevent_initial_call=True,
)
def download_reporting_csv(n_clicks, records):
    if not n_clicks:
        raise PreventUpdate

    df = normalize_records_to_df(records)
    if df.empty:
        raise PreventUpdate

    df_out = df.copy()
    df_out["test_date"] = df_out["test_date"].astype(str)
    return dict(
        content=df_out.to_csv(index=False),
        filename="warehouse_step_test_reporting.csv",
        type="text/csv",
    )

