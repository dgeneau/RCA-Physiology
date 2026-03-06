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


cfg = WarehouseAPIConfig(base_url=SITE_URL)
wc = WarehouseClient(cfg, token_getter=auth.get_token)

dash.register_page(__name__, path="/reports", name="Reporting")


# =========================================================
# CONSTANTS / DEFAULTS
# =========================================================
REPORT_TABLE_COLUMNS = [
    {"name": "Test Date", "id": "test_date"},
    {"name": "Athlete ID", "id": "profile_id"},
    {"name": "Session ID", "id": "session_id"},
    {"name": "Step", "id": "step_no"},
    {"name": "Step Type", "id": "step_type"},
    {"name": "Mode", "id": "mode"},
    {"name": "Test Type", "id": "test_type"},
    {"name": "Target PO", "id": "target_po_w"},
    {"name": "Actual PO", "id": "actual_po_w"},
    {"name": "HR", "id": "hr_bpm"},
    {"name": "La", "id": "lactate_mmol"},
    {"name": "VO2", "id": "vo2"},
    {"name": "Rate", "id": "rate_spm"},
    {"name": "Split (s/500)", "id": "split_sec_per_500"},
    {"name": "RPE", "id": "rpe"},
    {"name": "Time (s)", "id": "time_s"},
    {"name": "Body Mass", "id": "body_mass_kg"},
    {"name": "Notes", "id": "notes"},
]


# =========================================================
# HELPERS
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

def _extract_record_payload(rec):
    """
    Try to pull the actual ingested payload out of a warehouse record.
    Supports a few likely response shapes.
    """
    if not isinstance(rec, dict):
        return {}

    # Case 1: raw fields are already top-level
    if "profile_id" in rec or "session_id" in rec or "test_date" in rec:
        return rec

    # Case 2: warehouse wraps payload under 'data'
    if isinstance(rec.get("data"), dict):
        return rec["data"]

    # Case 3: warehouse wraps payload under 'record'
    if isinstance(rec.get("record"), dict):
        return rec["record"]

    # Case 4: warehouse wraps payload under 'raw'
    if isinstance(rec.get("raw"), dict):
        return rec["raw"]

    # Fallback: return original and let downstream handle it
    return rec


def normalize_records_to_df(records):
    if not records:
        return pd.DataFrame()

    extracted = [_extract_record_payload(r) for r in records]
    df = pd.DataFrame(extracted)

    expected_cols = [
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
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None

    numeric_cols = [
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
    ]
    for c in numeric_cols:
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
        dff = dff[dff["test_date"] <= pd.to_datetime(end_date)]

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
    Read records from the warehouse using WarehouseClient.list_records().
    Note: collected_after/before filters operate on warehouse collection time,
    not necessarily your test_date, so we still apply local filtering after load.
    """
    if not source_uuid:
        raise ValueError("VO2_STEP_SOURCE_UUID is not set.")

    collected_after = pd.to_datetime(start_date) if start_date else None
    collected_before = pd.to_datetime(end_date) if end_date else None

    records = wc.list_records(
        source_uuid=source_uuid,
        subject=int(profile_id) if profile_id not in (None, "", []) else None,
        collected_after=collected_after,
        collected_before=collected_before,
        role="primary",
        page_size=500,
    )

    df = normalize_records_to_df(records)
    df = apply_local_filters(
        df,
        profile_id=profile_id,
        start_date=start_date,
        end_date=end_date,
        test_type=test_type,
        mode=mode,
    )
    return df

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
        html.H2("Reporting"),
        html.Div("Pull step test data from the warehouse and review athlete trends."),
        html.Hr(),

        dcc.Store(id="reporting-athlete-options-store"),
        dcc.Store(id="reporting-data-store"),

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
                                                date=(date.today() - timedelta(days=180)).isoformat(),
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
                                                date=date.today().isoformat(),
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
                            dcc.Download(id="reporting-download"),
                            html.Hr(),
                            dbc.Alert(id="reporting-status-msg", is_open=False),
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
                                columns=REPORT_TABLE_COLUMNS,
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

        records = df.to_dict("records")
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
)
def update_reporting_table(records):
    df = normalize_records_to_df(records)

    if df.empty:
        return [], "0", "0", "—", "—"

    df_display = df.copy()
    df_display["test_date"] = df_display["test_date"].astype(str)
    df_display["split_sec_per_500"] = df_display["split_sec_per_500"].apply(format_split_mmss)

    rows_txt = str(len(df_display))
    sessions_txt = str(df_display["session_id"].dropna().nunique())

    avg_po = df["actual_po_w"].mean(skipna=True)
    avg_hr = df["hr_bpm"].mean(skipna=True)

    avg_po_txt = f"{avg_po:.1f} W" if pd.notna(avg_po) else "—"
    avg_hr_txt = f"{avg_hr:.1f} bpm" if pd.notna(avg_hr) else "—"

    return df_display.to_dict("records"), rows_txt, sessions_txt, avg_po_txt, avg_hr_txt


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
    df = add_athlete_names(df, athlete_options)

    # ---------- Empty figs ----------
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        title="No data",
    )

    if df.empty:
        return empty_fig, empty_fig, empty_fig

    # ---------- PO vs HR ----------
    df_hr = df.dropna(subset=["actual_po_w", "hr_bpm"]).copy()
    if df_hr.empty:
        fig_hr = empty_fig
    else:
        fig_hr = px.scatter(
            df_hr,
            x="actual_po_w",
            y="hr_bpm",
            color="athlete_name",
            hover_data=["test_date", "step_no", "test_type", "mode"],
            title="Heart Rate vs Actual PO",
            labels={
                "actual_po_w": "Actual PO (W)",
                "hr_bpm": "Heart Rate (bpm)",
                "athlete_name": "Athlete",
            },
        )
        fig_hr.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))

    # ---------- PO vs La ----------
    df_la = df.dropna(subset=["actual_po_w", "lactate_mmol"]).copy()
    if df_la.empty:
        fig_la = empty_fig
    else:
        fig_la = px.scatter(
            df_la,
            x="actual_po_w",
            y="lactate_mmol",
            color="athlete_name",
            hover_data=["test_date", "step_no", "test_type", "mode"],
            title="Blood Lactate vs Actual PO",
            labels={
                "actual_po_w": "Actual PO (W)",
                "lactate_mmol": "Blood Lactate",
                "athlete_name": "Athlete",
            },
        )
        fig_la.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))

    # ---------- Session trend ----------
    # one point per session: max PO + max HR + max La
    sess = df.copy()
    sess["test_date_dt"] = pd.to_datetime(sess["test_date"], errors="coerce")

    agg = (
        sess.groupby(["session_id", "profile_id", "athlete_name", "test_date_dt"], dropna=False)
        .agg(
            max_po=("actual_po_w", "max"),
            max_hr=("hr_bpm", "max"),
            max_la=("lactate_mmol", "max"),
            avg_po=("actual_po_w", "mean"),
            avg_hr=("hr_bpm", "mean"),
        )
        .reset_index()
        .sort_values("test_date_dt")
    )

    if agg.empty:
        fig_trend = empty_fig
    else:
        fig_trend = px.line(
            agg,
            x="test_date_dt",
            y="max_po",
            color="athlete_name",
            markers=True,
            hover_data=["session_id", "max_hr", "max_la", "avg_po", "avg_hr"],
            title="Session Trend: Max PO by Test Date",
            labels={
                "test_date_dt": "Test Date",
                "max_po": "Max PO (W)",
                "athlete_name": "Athlete",
            },
        )
        fig_trend.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))

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