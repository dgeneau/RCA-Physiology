# erg_editor.py
"""The Erg Scores tab of the Reporting page.

Erg results are pushed from the Entry page one batch at a time, and until now
the only way to fix a mistyped rate or power was to ask someone with warehouse
access. This module pulls those records back out, puts them in an editable
table, and writes the changed cells back to the same records -- the same
load / edit / review / update flow the step-test tab uses, against the erg
data source instead.

It lives outside pages/ because it is a piece of the reporting page rather than
a page of its own; pages/reports.py imports it and drops the layout into a tab.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

import dash
from dash import html, dcc, dash_table, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from auth_setup import auth
from erg_protocols import (
    ERG_PROTOCOL_OPTIONS,
    ERG_PROTOCOL_VALUES,
    POWER_ONLY_PROTOCOLS,
    infer_protocol,
    protocol_label,
)
from settings import SITE_URL, ERG_TEST_SOURCE_UUID
from warehouse import WarehouseAPIConfig, WarehouseClient, WarehouseClientError


cfg = WarehouseAPIConfig(base_url=SITE_URL)
wc = WarehouseClient(cfg, token_getter=auth.get_token)


# The athlete dropdown options are fetched once by pages/reports.py; this tab
# reads the same store rather than pulling the profile list a second time.
ATHLETE_OPTIONS_STORE = "reporting-athlete-options-store"


# =========================================================
# CONSTANTS
# =========================================================
# The fields as they are stored in the warehouse, in payload order.
ERG_DATA_COLUMNS = [
    "row_no",
    "profile_id",
    "test_date",
    "protocol",
    "distance_m",
    "stroke_rate_spm",
    "power_w",
    "time_min",
    "time_s",
]

ERG_TABLE_COLUMNS = [
    {"name": "Record UUID", "id": "__record_uuid", "editable": False},
    {"name": "Athlete", "id": "athlete_name", "editable": False},
    {"name": "Athlete ID", "id": "profile_id", "editable": False},
    {"name": "Test Date", "id": "test_date", "editable": True},
    {"name": "Test", "id": "protocol", "editable": True, "presentation": "dropdown"},
    {"name": "Distance (m)", "id": "distance_m", "editable": True},
    {"name": "Stroke Rate (spm)", "id": "stroke_rate_spm", "editable": True},
    {"name": "Power (W)", "id": "power_w", "editable": True},
    {"name": "Time (min)", "id": "time_min", "editable": True},
    {"name": "Time (s)", "id": "time_s", "editable": True},
    {"name": "Total Time", "id": "__total_time", "editable": False},
    {"name": "Split /500", "id": "__split_500", "editable": False},
    {"name": "Row", "id": "row_no", "editable": False},
]

ERG_EDITABLE_COLUMNS = {
    col["id"]
    for col in ERG_TABLE_COLUMNS
    if col.get("editable") and not col["id"].startswith("__")
}

ERG_NUMERIC_COLUMNS = {
    "row_no",
    "profile_id",
    "distance_m",
    "stroke_rate_spm",
    "power_w",
    "time_min",
    "time_s",
}

ERG_INTEGER_COLUMNS = {"row_no", "profile_id", "distance_m"}

# Entry writes 0.1 into a numeric erg field it has no value for (see
# coerce_erg_positive_number in pages/entry.py), including the minutes or
# seconds part of a time that lands exactly on a minute. Showing 0.1 back to a
# practitioner would read as data; this tab displays those cells blank and
# writes the sentinel again on save, so a round trip through the editor leaves
# an untouched record byte-identical.
ERG_MISSING_SENTINEL = 0.1
ERG_SENTINEL_COLUMNS = {"stroke_rate_spm", "power_w", "time_min", "time_s"}

# Bounds are advisory: they catch a slipped decimal point, not an unusual
# athlete. Anything outside them blocks the update until it is corrected.
ERG_NUMERIC_RANGES = {
    "distance_m": (100, 50000, "Distance should be between 100 and 50,000 m."),
    "stroke_rate_spm": (10, 60, "Stroke rate should be between 10 and 60 spm."),
    "power_w": (20, 1200, "Power should be between 20 and 1200 W."),
    "time_min": (0, 600, "Time (min) should be between 0 and 600."),
    "time_s": (0, 60, "Time (s) is the seconds part -- use 0 to 59.99."),
}

ERG_PROTOCOL_FILTER_OPTIONS = [{"label": "All", "value": "all"}] + ERG_PROTOCOL_OPTIONS


# =========================================================
# SMALL HELPERS
# =========================================================
def make_card(title, body):
    return dbc.Card(
        [dbc.CardHeader(html.B(title)), dbc.CardBody(body)],
        className="shadow-sm",
    )


def to_float(x):
    try:
        if x is None or x == "":
            return None
        value = float(x)
        return None if pd.isna(value) else value
    except Exception:
        return None


def safe_date_str(x):
    if x in (None, ""):
        return None
    try:
        parsed = pd.to_datetime(x, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def format_mmss(total_seconds):
    """Seconds as m:ss.s, the way an erg score is read aloud."""
    value = to_float(total_seconds)
    if value is None or value <= 0:
        return "—"
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes}:{seconds:04.1f}"


def erg_total_seconds(row):
    """Total time for a row, treating the missing-value sentinel as zero."""
    minutes = strip_sentinel(row.get("time_min"), "time_min")
    seconds = strip_sentinel(row.get("time_s"), "time_s")
    minutes = to_float(minutes) or 0.0
    seconds = to_float(seconds) or 0.0
    total = minutes * 60 + seconds
    return total if total > 0 else None


def erg_split_seconds(row):
    total = erg_total_seconds(row)
    distance = to_float(row.get("distance_m"))
    if total is None or not distance or distance <= 0:
        return None
    return total / (distance / 500.0)


def strip_sentinel(value, column_id):
    """The stored 0.1 placeholder, read back as 'no value'."""
    if column_id not in ERG_SENTINEL_COLUMNS:
        return value
    number = to_float(value)
    if number is not None and abs(number - ERG_MISSING_SENTINEL) < 1e-9:
        return None
    return value


def clean_cell_value(value, column_id):
    """A table cell as it should be stored in the warehouse."""
    if isinstance(value, np.generic):
        value = value.item()

    if value is not None and not isinstance(value, (list, dict)):
        try:
            if pd.isna(value):
                value = None
        except (TypeError, ValueError):
            pass

    if value == "":
        value = None

    if column_id in ERG_SENTINEL_COLUMNS and value is None:
        return ERG_MISSING_SENTINEL

    if value is None:
        return None

    if column_id in ERG_NUMERIC_COLUMNS:
        number = to_float(value)
        if number is None:
            return None
        if column_id in ERG_INTEGER_COLUMNS:
            return int(number)
        return number

    if column_id == "test_date":
        return safe_date_str(value)

    return value


def values_equal(a, b, column_id):
    a_clean = clean_cell_value(a, column_id)
    b_clean = clean_cell_value(b, column_id)

    if a_clean is None and b_clean is None:
        return True
    if a_clean is None or b_clean is None:
        return False

    if column_id in ERG_NUMERIC_COLUMNS:
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
    except (TypeError, ValueError):
        pass
    if column_id in ERG_SENTINEL_COLUMNS and strip_sentinel(value, column_id) is None:
        return "—"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def row_label(row):
    athlete = row.get("athlete_name") or row.get("profile_id") or "Athlete"
    test_date = display_value(row.get("test_date"))
    piece = protocol_label(row.get("protocol"))
    distance = display_value(row.get("distance_m"), "distance_m")
    return f"{athlete} | {test_date} | {piece} | {distance} m"


def row_to_warehouse_payload(row):
    return {col: clean_cell_value(row.get(col), col) for col in ERG_DATA_COLUMNS}


# =========================================================
# WAREHOUSE READ
# =========================================================
def extract_erg_payload(rec):
    """The ingested payload out of a warehouse record wrapper."""
    if not isinstance(rec, dict):
        return {}

    # Already unwrapped (rows coming back out of the dcc.Store).
    if "__record_uuid" in rec or "distance_m" in rec:
        return rec

    for key in ("data", "record", "raw"):
        if isinstance(rec.get(key), dict):
            payload = rec[key].copy()
            payload["__record_uuid"] = rec.get("uuid")
            payload["__dataset_uuid"] = rec.get("dataset_uuid")
            return payload

    return rec


def normalize_erg_records_to_df(records):
    expected_cols = ["__record_uuid", "__dataset_uuid"] + ERG_DATA_COLUMNS

    if not records:
        return pd.DataFrame(columns=expected_cols)

    df = pd.DataFrame([extract_erg_payload(r) for r in records])

    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    for col in ERG_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce")

    # Records written before the protocol field existed get one here rather
    # than on the way out, so a back-filled value is part of the row the editor
    # compares against and does not show up as an edit nobody made.
    df["protocol"] = [
        row["protocol"]
        if row.get("protocol") in ERG_PROTOCOL_VALUES
        else infer_protocol(row.get("distance_m"), erg_total_seconds(row))
        for row in df.to_dict("records")
    ]

    return df


def apply_erg_filters(df, profile_id=None, start_date=None, end_date=None, protocol=None):
    if df is None or df.empty:
        return df

    dff = df.copy()

    if profile_id not in (None, "", []):
        dff = dff[dff["profile_id"] == pd.to_numeric(profile_id, errors="coerce")]

    if start_date:
        dff = dff[dff["test_date"] >= pd.to_datetime(start_date)]

    if end_date:
        dff = dff[dff["test_date"] < (pd.to_datetime(end_date) + pd.Timedelta(days=1))]

    if protocol not in (None, "", "all"):
        dff = dff[dff["protocol"] == protocol]

    return dff.sort_values(
        by=["test_date", "profile_id", "protocol", "row_no"],
        ascending=[False, True, True, True],
        na_position="last",
    )


def fetch_erg_data_from_warehouse(profile_id=None, start_date=None, end_date=None, protocol=None):
    if not ERG_TEST_SOURCE_UUID:
        raise ValueError("ERG_TEST_SOURCE_UUID is not set.")

    records = wc.list_records(
        source_uuid=ERG_TEST_SOURCE_UUID,
        subject=int(profile_id) if profile_id not in (None, "", []) else None,
        role="primary",
        page_size=500,
    )

    df = normalize_erg_records_to_df(records)
    return apply_erg_filters(
        df,
        profile_id=profile_id,
        start_date=start_date,
        end_date=end_date,
        protocol=protocol,
    )


def dataframe_to_store_records(df):
    if df is None or df.empty:
        return []

    out = df.copy().replace({np.nan: None})
    if "test_date" in out.columns:
        out["test_date"] = out["test_date"].apply(
            lambda x: x.isoformat() if hasattr(x, "isoformat") and pd.notna(x) else x
        )
    return out.to_dict("records")


def add_athlete_names(df, athlete_options):
    dff = df.copy()
    id_to_name = {}
    for opt in athlete_options or []:
        try:
            id_to_name[int(opt["value"])] = opt["label"]
        except (KeyError, TypeError, ValueError):
            continue

    dff["athlete_name"] = dff["profile_id"].map(id_to_name).fillna(
        dff["profile_id"].astype("Int64").astype(str)
    )
    return dff


def records_to_table_rows(records, athlete_options):
    """Store records as the rows the editable table shows."""
    df = normalize_erg_records_to_df(records)
    if df.empty:
        return []

    df = add_athlete_names(df, athlete_options)
    rows = []
    for row in df.to_dict("records"):
        display_row = dict(row)
        display_row["test_date"] = safe_date_str(row.get("test_date"))
        for col in ERG_SENTINEL_COLUMNS:
            display_row[col] = strip_sentinel(row.get(col), col)
        display_row["__total_time"] = format_mmss(erg_total_seconds(row))
        display_row["__split_500"] = format_mmss(erg_split_seconds(row))
        rows.append(display_row)
    return rows


# =========================================================
# EDIT / VALIDATION
# =========================================================
def get_changed_cells(original_records, edited_rows):
    original_df = normalize_erg_records_to_df(original_records)
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

        for col in ERG_EDITABLE_COLUMNS:
            original_value = original_row.get(col)
            if col == "test_date":
                original_value = safe_date_str(original_value)
            if values_equal(edited_row.get(col), original_value, col):
                continue
            changes.append(
                {
                    "record_uuid": str(record_uuid),
                    "column_id": col,
                    "column_name": next(
                        (c["name"] for c in ERG_TABLE_COLUMNS if c["id"] == col), col
                    ),
                    "old": clean_cell_value(original_value, col),
                    "new": clean_cell_value(edited_row.get(col), col),
                    "row_label": row_label(edited_row),
                }
            )

    return changes


def validate_erg_rows(edited_rows):
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

        if safe_date_str(row.get("test_date")) is None:
            add_issue("test_date", "Test date must be a real date (YYYY-MM-DD).")

        power_only = row.get("protocol") in POWER_ONLY_PROTOCOLS

        if power_only:
            if to_float(strip_sentinel(row.get("power_w"), "power_w")) is None:
                add_issue("power_w", "Power is required for a peak power test.")
        elif to_float(row.get("distance_m")) is None:
            add_issue("distance_m", "Distance is required.")

        if row.get("protocol") not in ERG_PROTOCOL_VALUES:
            add_issue(
                "protocol",
                "Test must be one of "
                + ", ".join(protocol_label(p) for p in ERG_PROTOCOL_VALUES)
                + ".",
            )

        for col, (low, high, message) in ERG_NUMERIC_RANGES.items():
            raw_value = strip_sentinel(row.get(col), col)
            if raw_value in (None, ""):
                continue
            value = to_float(raw_value)
            if value is None:
                add_issue(col, f"{col} must be numeric.")
            elif value < low or value > high:
                add_issue(col, message)

        if erg_total_seconds(row) is None and not power_only:
            add_issue("time_min", "Enter a time as minutes and/or seconds.")

    return issues


def build_change_summary(changes, issues):
    if issues:
        issue_rows = [html.Li(f"{i['row_label']} - {i['message']}") for i in issues[:12]]
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


def build_table_styles(changes, issues):
    styles = [
        {"if": {"column_id": col}, "backgroundColor": "#f3f9ff"}
        for col in ERG_EDITABLE_COLUMNS
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


def build_table_columns(edit_mode=False):
    columns = []
    for col in ERG_TABLE_COLUMNS:
        next_col = col.copy()
        if next_col["id"] in ERG_EDITABLE_COLUMNS:
            next_col["editable"] = bool(edit_mode)
        columns.append(next_col)
    return columns


# =========================================================
# LAYOUT
# =========================================================
layout = dbc.Container(
    [
        dcc.Store(id="erg-report-data-store"),
        dcc.Store(id="erg-report-edit-mode-store", data=False),

        dbc.Row(
            [
                dbc.Col(
                    make_card(
                        "Filters",
                        [
                            dbc.Label("Athlete"),
                            dcc.Dropdown(
                                id="erg-report-athlete",
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
                                            dbc.Label("Start Date", html_for="erg-report-start-date"),
                                            dbc.Input(
                                                type="date",
                                                size="sm",
                                                id="erg-report-start-date",
                                                value=(date.today() - timedelta(days=365)).isoformat(),
                                            ),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("End Date", html_for="erg-report-end-date"),
                                            dbc.Input(
                                                type="date",
                                                size="sm",
                                                id="erg-report-end-date",
                                                value=(date.today() + timedelta(days=365)).isoformat(),
                                            ),
                                        ],
                                        md=6,
                                    ),
                                ],
                                className="g-2",
                            ),
                            html.Br(),

                            dbc.Label("Test"),
                            dcc.Dropdown(
                                id="erg-report-protocol",
                                options=ERG_PROTOCOL_FILTER_OPTIONS,
                                value="all",
                                clearable=False,
                            ),
                            html.Br(),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            "Load Erg Scores",
                                            id="erg-report-load-btn",
                                            color="primary",
                                            className="w-100",
                                        ),
                                        md=6,
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Download CSV",
                                            id="erg-report-download-btn",
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
                                id="erg-report-edit-btn",
                                color="secondary",
                                outline=True,
                                className="w-100",
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        "Revert Changes",
                                        id="erg-report-revert-btn",
                                        color="secondary",
                                        outline=True,
                                        className="w-100",
                                        disabled=True,
                                    ),
                                    dbc.Button(
                                        "Review Changes",
                                        id="erg-report-update-btn",
                                        color="success",
                                        className="w-100 mt-2",
                                        disabled=True,
                                    ),
                                ],
                                className="mt-2",
                            ),
                            dcc.Download(id="erg-report-download"),
                            html.Hr(),
                            html.Small(
                                "Load a range, switch on Edit Data, correct the tinted cells, "
                                "then review the changes before they reach the warehouse. The "
                                "athlete column stays read-only. On a 30 min piece the distance "
                                "is the result, so it is editable like any other measurement; "
                                "results recorded before the Test column existed are read as a "
                                "2000 m or 6000 m from their distance.",
                                className="text-muted",
                            ),
                            html.Hr(),
                            dbc.Alert(id="erg-report-status-msg", is_open=False),
                            dbc.Alert(id="erg-report-update-msg", is_open=False),
                        ],
                    ),
                    md=3,
                ),

                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(make_card("Rows", html.H4(id="erg-report-rows", className="m-0")), md=3),
                                dbc.Col(make_card("Athletes", html.H4(id="erg-report-athletes", className="m-0")), md=3),
                                dbc.Col(make_card("Avg Split", html.H4(id="erg-report-avg-split", className="m-0")), md=3),
                            ],
                            className="g-2 mb-3",
                        ),

                        make_card(
                            "Erg Results",
                            dash_table.DataTable(
                                id="erg-report-table",
                                data=[],
                                columns=build_table_columns(False),
                                hidden_columns=["__record_uuid", "profile_id"],
                                dropdown={"protocol": {"options": ERG_PROTOCOL_OPTIONS}},
                                editable=True,
                                page_action="native",
                                page_size=15,
                                sort_action="native",
                                filter_action="native",
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
                                    for col in ERG_EDITABLE_COLUMNS
                                ],
                                style_data_conditional=build_table_styles([], []),
                            ),
                        ),
                    ],
                    md=9,
                ),
            ],
            className="g-3",
        ),

        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Review Erg Score Changes")),
                dbc.ModalBody(id="erg-report-change-summary"),
                dbc.ModalFooter(
                    [
                        dbc.Button("Cancel", id="erg-report-cancel-update-btn", color="secondary", outline=True),
                        dbc.Button(
                            "Update Warehouse",
                            id="erg-report-confirm-update-btn",
                            color="success",
                            disabled=True,
                        ),
                    ]
                ),
            ],
            id="erg-report-review-modal",
            size="lg",
            is_open=False,
            scrollable=True,
        ),
    ],
    fluid=True,
    class_name="px-0",
)


# =========================================================
# CALLBACKS
# =========================================================
@dash.callback(
    Output("erg-report-athlete", "options"),
    Input(ATHLETE_OPTIONS_STORE, "data"),
)
def apply_erg_athlete_options(options):
    return options or []


@dash.callback(
    Output("erg-report-data-store", "data"),
    Output("erg-report-status-msg", "children"),
    Output("erg-report-status-msg", "color"),
    Output("erg-report-status-msg", "is_open"),
    Input("erg-report-load-btn", "n_clicks"),
    State("erg-report-athlete", "value"),
    State("erg-report-start-date", "value"),
    State("erg-report-end-date", "value"),
    State("erg-report-protocol", "value"),
    prevent_initial_call=True,
)
def load_erg_data(n_clicks, athlete_id, start_date, end_date, protocol):
    if not n_clicks:
        raise PreventUpdate

    try:
        df = fetch_erg_data_from_warehouse(
            profile_id=athlete_id,
            start_date=safe_date_str(start_date),
            end_date=safe_date_str(end_date),
            protocol=protocol,
        )

        if df.empty:
            return [], "No erg scores found for the selected filters.", "warning", True

        return (
            dataframe_to_store_records(df),
            f"Loaded {len(df)} erg row(s) from warehouse.",
            "success",
            True,
        )

    except (WarehouseClientError, ValueError, AttributeError) as e:
        return [], f"Load failed: {e}", "danger", True
    except Exception as e:
        return [], f"Unexpected error: {e}", "danger", True


@dash.callback(
    Output("erg-report-table", "data"),
    Output("erg-report-rows", "children"),
    Output("erg-report-athletes", "children"),
    Output("erg-report-avg-split", "children"),
    Input("erg-report-data-store", "data"),
    Input(ATHLETE_OPTIONS_STORE, "data"),
)
def update_erg_table(records, athlete_options):
    rows = records_to_table_rows(records, athlete_options)
    if not rows:
        return [], "0", "0", "—"

    df = pd.DataFrame(rows)


    splits = [erg_split_seconds(row) for row in rows]
    splits = [s for s in splits if s is not None]
    avg_split = float(np.mean(splits)) if splits else None

    return (
        rows,
        str(len(rows)),
        str(df["profile_id"].dropna().nunique()),
        f"{format_mmss(avg_split)} /500" if avg_split else "—",
    )


@dash.callback(
    Output("erg-report-edit-mode-store", "data"),
    Input("erg-report-edit-btn", "n_clicks"),
    Input("erg-report-data-store", "data"),
    State("erg-report-edit-mode-store", "data"),
)
def toggle_erg_edit_mode(edit_clicks, records, edit_mode):
    trigger = ctx.triggered_id
    if trigger == "erg-report-edit-btn":
        return not bool(edit_mode)
    if trigger == "erg-report-data-store":
        return False
    return bool(edit_mode)


@dash.callback(
    Output("erg-report-table", "columns"),
    Output("erg-report-table", "style_data_conditional"),
    Output("erg-report-edit-btn", "children"),
    Output("erg-report-edit-btn", "color"),
    Output("erg-report-revert-btn", "disabled"),
    Output("erg-report-update-btn", "disabled"),
    Input("erg-report-edit-mode-store", "data"),
    Input("erg-report-table", "data"),
    State("erg-report-data-store", "data"),
)
def update_erg_edit_controls(edit_mode, table_rows, original_records):
    changes = get_changed_cells(original_records, table_rows)
    issues = validate_erg_rows(table_rows) if changes else []
    styles = build_table_styles(changes, issues)
    edit_mode = bool(edit_mode)
    has_changes = bool(changes)

    return (
        build_table_columns(edit_mode),
        styles,
        "Exit Edit Mode" if edit_mode else "Edit Data",
        "warning" if edit_mode else "secondary",
        not (edit_mode and has_changes),
        not (edit_mode and has_changes),
    )


@dash.callback(
    Output("erg-report-table", "data", allow_duplicate=True),
    Input("erg-report-revert-btn", "n_clicks"),
    State("erg-report-data-store", "data"),
    State(ATHLETE_OPTIONS_STORE, "data"),
    prevent_initial_call=True,
)
def revert_erg_changes(n_clicks, records, athlete_options):
    if not n_clicks:
        raise PreventUpdate
    return records_to_table_rows(records, athlete_options)


@dash.callback(
    Output("erg-report-review-modal", "is_open"),
    Output("erg-report-change-summary", "children"),
    Output("erg-report-confirm-update-btn", "disabled"),
    Input("erg-report-update-btn", "n_clicks"),
    Input("erg-report-cancel-update-btn", "n_clicks"),
    State("erg-report-table", "data"),
    State("erg-report-data-store", "data"),
    prevent_initial_call=True,
)
def toggle_erg_review_modal(review_clicks, cancel_clicks, table_rows, original_records):
    trigger = ctx.triggered_id
    if trigger == "erg-report-cancel-update-btn":
        return False, no_update, True

    if trigger != "erg-report-update-btn":
        raise PreventUpdate

    changes = get_changed_cells(original_records, table_rows)
    changed_uuids = {change["record_uuid"] for change in changes}
    # Only the rows being written are held to the bounds; an odd row someone
    # else entered years ago should not block today's correction.
    issues = [
        issue
        for issue in validate_erg_rows(table_rows)
        if issue["record_uuid"] in changed_uuids
    ]
    return True, build_change_summary(changes, issues), bool(issues or not changes)


def write_erg_record(record_uuid, patch_payload):
    """Push one edited erg record back, PATCH first and PUT as a fallback.

    Returns the UUID the row now lives under: the same one on success, or a new
    one when the deployment refuses in-place updates and the record has to be
    re-ingested and the old copy removed.
    """
    try:
        wc.patch_record(record_uuid=record_uuid, data=patch_payload)
        return record_uuid, None
    except WarehouseClientError as exc:
        # Python clears the `as` name at the end of the block, so the reason is
        # kept here for the combined message further down.
        patch_error = exc

    try:
        wc.put_record(record_uuid=record_uuid, data=patch_payload)
        return record_uuid, None
    except WarehouseClientError as exc:
        put_error = exc

    # Last resort: re-ingest the corrected row, then drop the stale one. The
    # delete is what keeps a failed in-place update from silently doubling the
    # athlete's result.
    try:
        dataset, created = wc.ingest_raw(
            source_uuid=ERG_TEST_SOURCE_UUID,
            records=[patch_payload["data"]],
            subject_field="profile_id",
            validate_client_side=False,
        )
        if created != 1:
            raise WarehouseClientError(f"Ingestion created {created} records instead of 1.")
        replacement = wc.list_records(
            source_uuid=ERG_TEST_SOURCE_UUID,
            role="primary",
            page_size=10,
            extra_params={"dataset_uuid": dataset.get("uuid")},
        )
        if not replacement:
            raise WarehouseClientError(
                "Ingestion succeeded but the replacement record could not be reloaded."
            )
    except WarehouseClientError as ingest_error:
        raise WarehouseClientError(
            f"PATCH, PUT and re-ingestion all failed for record {record_uuid}. "
            f"PATCH: {patch_error}. PUT: {put_error}. INGEST: {ingest_error}."
        ) from ingest_error

    replacement_record = replacement[0]
    try:
        wc.delete_record(record_uuid=record_uuid)
    except WarehouseClientError as delete_error:
        raise WarehouseClientError(
            "The corrected row was written, but removing the old copy failed, so the "
            f"athlete now has two. Old record: {record_uuid}. New record: "
            f"{replacement_record.get('uuid')}. DELETE: {delete_error}."
        ) from delete_error

    return replacement_record.get("uuid"), replacement_record


@dash.callback(
    Output("erg-report-data-store", "data", allow_duplicate=True),
    Output("erg-report-update-msg", "children"),
    Output("erg-report-update-msg", "color"),
    Output("erg-report-update-msg", "is_open"),
    Output("erg-report-review-modal", "is_open", allow_duplicate=True),
    Input("erg-report-confirm-update-btn", "n_clicks"),
    State("erg-report-table", "data"),
    State("erg-report-data-store", "data"),
    prevent_initial_call=True,
)
def update_erg_warehouse_records(n_clicks, edited_rows, original_records):
    if not n_clicks:
        raise PreventUpdate

    original_df = normalize_erg_records_to_df(original_records)
    edited_df = pd.DataFrame(edited_rows or [])

    if original_df.empty or edited_df.empty:
        return no_update, "No erg rows loaded to update.", "warning", True, False

    if "__record_uuid" not in original_df.columns or "__record_uuid" not in edited_df.columns:
        return (
            no_update,
            "Update failed: warehouse record UUIDs are missing. Reload the data and try again.",
            "danger",
            True,
            False,
        )

    changes = get_changed_cells(original_records, edited_rows)
    changed_uuids = {change["record_uuid"] for change in changes}
    issues = [i for i in validate_erg_rows(edited_rows) if i["record_uuid"] in changed_uuids]
    if issues:
        return no_update, "Update blocked: resolve validation issues before saving.", "danger", True, True

    original_by_uuid = {
        str(row["__record_uuid"]): row
        for row in original_df.to_dict("records")
        if row.get("__record_uuid")
    }

    changed_cols_by_uuid = {}
    for change in changes:
        changed_cols_by_uuid.setdefault(change["record_uuid"], set()).add(change["column_id"])

    updated_count = 0
    local_updates = {}

    try:
        for edited_row in edited_df.to_dict("records"):
            record_uuid = edited_row.get("__record_uuid")
            if not record_uuid:
                continue

            record_uuid = str(record_uuid)
            changed_cols = changed_cols_by_uuid.get(record_uuid)
            if not changed_cols:
                continue

            original_row = original_by_uuid.get(record_uuid)
            if not original_row:
                continue

            updated_row = original_row.copy()
            updated_row["test_date"] = safe_date_str(original_row.get("test_date"))
            for col in changed_cols:
                updated_row[col] = clean_cell_value(edited_row.get(col), col)

            patch_payload = {"data": row_to_warehouse_payload(updated_row)}
            if original_row.get("__dataset_uuid"):
                patch_payload["dataset"] = original_row["__dataset_uuid"]

            new_uuid, replacement_record = write_erg_record(record_uuid, patch_payload)

            updated_row["__record_uuid"] = str(new_uuid or record_uuid)
            if replacement_record:
                updated_row["__dataset_uuid"] = replacement_record.get("dataset_uuid") or \
                    original_row.get("__dataset_uuid")

            local_updates[record_uuid] = updated_row
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
        f"Updated {updated_count} erg record(s).",
        "success",
        True,
        False,
    )


@dash.callback(
    Output("erg-report-download", "data"),
    Input("erg-report-download-btn", "n_clicks"),
    State("erg-report-data-store", "data"),
    State(ATHLETE_OPTIONS_STORE, "data"),
    prevent_initial_call=True,
)
def download_erg_csv(n_clicks, records, athlete_options):
    if not n_clicks:
        raise PreventUpdate

    rows = records_to_table_rows(records, athlete_options)
    if not rows:
        raise PreventUpdate

    df = pd.DataFrame(rows).drop(columns=["__record_uuid", "__dataset_uuid"], errors="ignore")
    return dict(
        content=df.to_csv(index=False),
        filename="warehouse_erg_scores.csv",
        type="text/csv",
    )
