
# pages/entry.py


import os
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

import dash
from dash.dependencies import Input, Output, State
from dash import html, dcc, dash_table, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import json
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import numpy as np
import scipy as sp


from auth_setup import auth
from utils import fetch_options, fetch_profiles, fetch_profile

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
import io
import base64
from reportlab.lib.pagesizes import A4
import tempfile
from reportlab.lib.utils import ImageReader



# add near the top with other imports
from settings import SITE_URL, VO2_STEP_SOURCE_UUID  # TODO: define VO2_STEP_SOURCE_UUID in settings
from warehouse import WarehouseAPIConfig, WarehouseClient, WarehouseClientError

cfg = WarehouseAPIConfig(base_url=SITE_URL)
wc = WarehouseClient(cfg, token_getter=auth.get_token)


# Register this file as a page in the larger app
dash.register_page(__name__, path="/entry", name="Entry")




# ✅ Keys must match the "id" values
DEFAULT_ROWS = [{
    "step_no": 1,
    "Type": "Submax",
    "T_PO": None,
    "A_PO": None,
    "HR": None,
    "La": None,
    "V02": None,
    "rate": None,
    "split": None,   # computed
    "rpe": None,
},    
{"step_no": 2,
    "Type": "Submax",
    "T_PO": None,
    "A_PO": None,
    "HR": None,
    "La": None,
    "V02": None,
    "rate": None,
    "split": None,   # computed
    "rpe": None,
} , 
{"step_no": 3,
    "Type": "Submax",
    "T_PO": None,
    "A_PO": None,
    "HR": None,
    "La": None,
    "V02": None,
    "rate": None,
    "split": None,   # computed
    "rpe": None,
} 
]

TABLE_COLUMNS = [
    {"name": "Step Number", "id": "step_no", "type": "numeric"},
    {"name": "Submax/Max", "id": "Type", "type": "text"},
    {"name": "Target PO", "id": "T_PO", "type": "numeric"},
    {"name": "Actual PO", "id": "A_PO", "type": "numeric"},
    {"name": "Heart Rate", "id": "HR", "type": "numeric"},
    {"name": "Blood Lactate", "id": "La", "type": "numeric"},
    {"name": "V02", "id": "V02", "type": "numeric"},
    {"name": "Rate", "id": "rate", "type": "numeric"},

    # ✅ computed column (non-editable)
    {"name": "Split time", "id": "split", "type": "numeric", "editable": False},

    {"name": "RPE", "id": "rpe", "type": "numeric"},
]


#Zones Table
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
    {"name": "Split Low (s/500)", "id": "Split_low", "type": "numeric"},
    {"name": "Split High (s/500)", "id": "Split_high", "type": "numeric"},
    {"name": "Rate Low (spm)", "id": "Rate_low", "type": "numeric"},
    {"name": "Rate High (spm)", "id": "Rate_high", "type": "numeric"},
    {"name": "Notes", "id": "Notes", "type": "text"},
]


#Helpers

def make_card(title, body):
    return dbc.Card(
        [dbc.CardHeader(html.B(title)), dbc.CardBody(body)],
        className="shadow-sm",
    )

def safe_float(x):
    try:
        if x is None or x == "":
            return 0.0
        return float(x)
    except Exception:
        return 0.0

def to_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None

def estimate_split_seconds(power_w):
    """
    Returns estimated split (sec/500m) from power + (optional) rate.
    Base uses the common erg relationship split ~ power^(-1/3).
    Then a small rate adjustment is applied (tweak as you like).
    """
    p = to_float(power_w)

    if p is None or p <= 0:
        return None

    # Base pace from watts (sec/500m)
    pace = 500.0 * ((2.8 / p) ** (1.0 / 3.0))



    return round(pace, 2)

def add_poly_fit(fig, x, y, degree=2, name="Fit", color = 'red'):
    """
    Adds a polynomial fit line to an existing Plotly figure.
    Safely exits if not enough data.
    """
    if len(x) < degree + 1:
        return fig

    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)

    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = poly(x_fit)

    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            mode="lines",
            name=name,
            line=dict(dash="solid", color = color),
        )
    )
    return fig

def _first_numeric_in_rows(rows, keys):
    """Return first numeric value found scanning rows for any of the given keys."""
    rows = rows or []
    for r in rows:
        for k in keys:
            v = to_float(r.get(k))
            if v is not None:
                return float(v)
    return None

def hr_at_lactate(df_steps, target_la):
    """
    Estimate HR at a target lactate value using linear interpolation on La->HR.
    Returns None if insufficient data.
    """
    if df_steps is None or df_steps.empty:
        return None

    d = df_steps.copy()
    d["La"] = pd.to_numeric(d.get("La"), errors="coerce")
    d["HR"] = pd.to_numeric(d.get("HR"), errors="coerce")
    d = d.dropna(subset=["La", "HR"]).sort_values("La")

    if len(d) < 2:
        return None

    # Collapse duplicate lactate values (average HR for same La)
    d = d.groupby("La", as_index=False)["HR"].mean().sort_values("La")

    la = d["La"].to_numpy(dtype=float)
    hr = d["HR"].to_numpy(dtype=float)

    # If outside observed range, clamp to endpoints (safer than extrapolating wildly)
    if target_la <= la.min():
        return float(hr[0])
    if target_la >= la.max():
        return float(hr[-1])

    # Linear interpolation
    return float(np.interp(target_la, la, hr))

def build_zones_from_lt(hr_lt1, hr_lt2, band=5.0):
    """
    Build simple 5-zone model using HR at LT1 and LT2 with +/- band bpm around thresholds.
    """
    if hr_lt1 is None or hr_lt2 is None:
        return None

    # Ensure ordering
    hr_lt1 = float(hr_lt1)
    hr_lt2 = float(hr_lt2)
    if hr_lt2 < hr_lt1:
        hr_lt1, hr_lt2 = hr_lt2, hr_lt1

    z1_hi = max(0.0, hr_lt1 - band)
    z2_lo = z1_hi
    z2_hi = hr_lt1 + band
    z3_lo = z2_hi
    z3_hi = max(z3_lo, hr_lt2 - band)
    z4_lo = z3_hi
    z4_hi = hr_lt2 + band
    z5_lo = z4_hi

    zones = [
        {"Zone": "Z1", "HR_low": None,      "HR_high": round(z1_hi, 0), "Notes": "Easy / recovery"},
        {"Zone": "Z2", "HR_low": round(z2_lo, 0), "HR_high": round(z2_hi, 0), "Notes": "Endurance"},
        {"Zone": "Z3", "HR_low": round(z3_lo, 0), "HR_high": round(z3_hi, 0), "Notes": "Tempo"},
        {"Zone": "Z4", "HR_low": round(z4_lo, 0), "HR_high": round(z4_hi, 0), "Notes": "Threshold"},
        {"Zone": "Z5", "HR_low": round(z5_lo, 0), "HR_high": None,      "Notes": "VO₂ / high intensity"},
    ]
    return zones

def _interp_y_at_x(df, x_col, y_col, x_target):
    """
    Linear interpolation of y at x_target using df[x_col]->df[y_col].
    - sorts by x_col
    - averages duplicates
    - clamps outside range to endpoints
    """
    if df is None or df.empty or x_target is None:
        return None

    d = df.copy()
    d[x_col] = pd.to_numeric(d.get(x_col), errors="coerce")
    d[y_col] = pd.to_numeric(d.get(y_col), errors="coerce")
    d = d.dropna(subset=[x_col, y_col])
    if len(d) < 2:
        return None

    # collapse duplicates
    d = d.groupby(x_col, as_index=False)[y_col].mean().sort_values(x_col)

    x = d[x_col].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)

    if x_target <= x.min():
        return float(y[0])
    if x_target >= x.max():
        return float(y[-1])

    return float(np.interp(float(x_target), x, y))

def _zone_row_from_bounds(df_steps, zone_name, hr_low, hr_high, notes=""):
    """
    Build a zone row including HR, PO, Split, Rate bounds.
    PO/rate are interpolated vs HR; split is computed from PO.
    """
    po_low = _interp_y_at_x(df_steps, "HR", "A_PO", hr_low) if hr_low is not None else None
    po_high = _interp_y_at_x(df_steps, "HR", "A_PO", hr_high) if hr_high is not None else None

    rate_low = _interp_y_at_x(df_steps, "HR", "rate", hr_low) if hr_low is not None else None
    rate_high = _interp_y_at_x(df_steps, "HR", "rate", hr_high) if hr_high is not None else None

    split_low_sec = estimate_split_seconds(po_low) if po_low is not None else None
    split_high_sec = estimate_split_seconds(po_high) if po_high is not None else None

    split_low = format_split_mmss(split_low_sec)
    split_high = format_split_mmss(split_high_sec)

    return {
        "Zone": zone_name,
        "HR_low": round(hr_low, 0) if hr_low is not None else None,
        "HR_high": round(hr_high, 0) if hr_high is not None else None,
        "PO_low": round(po_low, 1) if po_low is not None else None,
        "PO_high": round(po_high, 1) if po_high is not None else None,
        "Split_low": split_low,
        "Split_high": split_high,
        "Rate_low": round(rate_low, 1) if rate_low is not None else None,
        "Rate_high": round(rate_high, 1) if rate_high is not None else None,
        "Notes": notes or "",
    }

def format_split_mmss(split_seconds):
    """
    Convert split in seconds to mm:ss.ms (e.g. 112.34 -> '1:52.34')
    """
    if split_seconds is None:
        return None

    try:
        total_seconds = float(split_seconds)
    except Exception:
        return None

    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:05.2f}"

def _interp_at_lactate(df, y_col, la_target):
    """
    Interpolate y_col at a given lactate value using La -> y_col.
    Clamps outside range to endpoints.
    """
    if df is None or df.empty or la_target is None:
        return None

    d = df.copy()
    d["La"] = pd.to_numeric(d.get("La"), errors="coerce")
    d[y_col] = pd.to_numeric(d.get(y_col), errors="coerce")
    d = d.dropna(subset=["La", y_col])
    if len(d) < 2:
        return None

    d = d.groupby("La", as_index=False)[y_col].mean().sort_values("La")
    la = d["La"].to_numpy(float)
    y = d[y_col].to_numpy(float)

    if la_target <= la.min():
        return float(y[0])
    if la_target >= la.max():
        return float(y[-1])

    return float(np.interp(float(la_target), la, y))



layout = dbc.Container(
    [
        html.H2("V02 Step Testing"),
        html.Div("Text inputs, numeric inputs, radio buttons, and an editable DataTable."),
        html.Hr(),

        dbc.Row(
            [
                dbc.Col(
                    make_card(
                        "Entry Fields",
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Athlete"),
                                            dcc.Dropdown(
                                                id="form-name",
                                                options = [],
                                                placeholder="select athlete for analysis",
                                                value=None,
                                            ),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Body Mass (kg)"),
                                            dbc.Input(
                                                id="form-mass",
                                                type="number",
                                                min = 0, 
                                                step = 0.1,
                                                placeholder=0.0,
                                                value=None,
                                            ),
                                        ],
                                        md=6,
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
                                            ),
                                        ],
                                        md=6,
                                    ),
                                ],
                                className="g-3",
                            ),
                            html.Br(),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Test Type"),
                                            dbc.RadioItems(
                                                id="form-status",
                                                options=[
                                                    {"label": "Erg C2", "value": "erg_C2"},
                                                    {"label": "Erg RP3", "value": "erg_RP3"},
                                                    {"label": "On-Water", "value": "row"},
                                                    {"label": "Bike", "value": "bike"},
                                                    {"label": "Other", "value": "other"},
                                                ],
                                                value="erg_C2",
                                                inline=True,
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Mode"),
                                            dbc.RadioItems(
                                                id="form-mode",
                                                options=[
                                                    {"label": "Max", "value": "Max"},
                                                    {"label": "Submax", "value": "Submax"},

                                                ],
                                                value="Submax",
                                                inline=True,
                                            ),
                                        ],
                                        md=4,
                                    ),
                                ],
                                className="g-3",
                            ),
                            html.Br(),
                            dbc.Label("Notes (multi-line)"),
                            dbc.Textarea(
                                id="form-notes",
                                placeholder="Anything you want to capture...",
                                value="",
                                style={"height": "110px"},
                            ),
                            html.Br(),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button("Submit", id="form-submit", color="primary", className="w-100"),
                                        md=4,
                                    ),
                                    dbc.Col(
                                        dbc.Button("Reset", id="form-reset", color="secondary", outline=True, className="w-100"),
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
                            html.Hr(),
                            dbc.Alert(id="form-status-msg", color="success", is_open=False),
                        ],
                    ),
                    md=4,
                ),

                dbc.Col(
                    make_card(
                        "Editable Items Table",
                        [
                            html.Div(
                                [
                                    dbc.Button("Add row", id="form-add-row", color="success", size="sm", className="me-2"),
                                    dbc.Button("Delete selected", id="form-delete-rows", color="danger", size="sm", outline=True),
                                ],
                                className="mb-2",
                            ),
                            dash_table.DataTable(
                                id="form-items-table",
                                data=DEFAULT_ROWS,
                                columns=TABLE_COLUMNS,
                                editable=True,
                                row_selectable="multi",
                                selected_rows=[],
                                page_action="native",
                                page_size=8,
                                style_table={"overflowX": "auto"},
                                style_cell={"padding": "8px", "fontFamily": "system-ui", "fontSize": 14},
                                style_header={"fontWeight": "600"},
                            ),
                            html.Br(),
                            dbc.Row(
                                [
                                    dbc.Col(make_card("Average P0", html.H4(id="form-avg-PO", className="m-0")), md=3),
                                    dbc.Col(make_card("Average HR", html.H4(id="form-avg-HR", className="m-0")), md=3),
                                    dbc.Col(make_card("Average Rate", html.H4(id="form-avg-rate", className="m-0")), md=3),
                                    dbc.Col(make_card("Step Count", html.H4(id="form-row-count", className="m-0")), md=3),
                                ],
                                className="g-2",
                            ),
                        ],
                    ),
                    md=8,
                ),
            ],
            className="g-3",
        ),

        html.Hr(),
        #make_card("Submission Preview", html.Pre(id="form-preview", className="m-0")),
        dcc.Store(id="form-last-payload"),
        ################
        html.Hr(),
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
            className="g-2",
        ),

        ################
        html.Hr(),
        make_card(
            "HR Training Zones (from Lactate Thresholds)",
            dash_table.DataTable(
                id="zones-table",
                data=ZONES_DEFAULT_ROWS,
                columns=ZONES_COLUMNS,
                editable=False,
                style_table={"overflowX": "auto"},
                style_cell={"padding": "8px", "fontFamily": "system-ui", "fontSize": 14},
                style_header={"fontWeight": "600"},
            ),
        ),
        ################
        # The Generate Report button triggers the generation of the report
        dcc.Store(id="report-generated", data=False),  # Track report generation status
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button("Download HR Zone Data", id="form-download-zones-btn", color="primary", outline=True, className="w-100"),
                    md=4,
                ),
            ],
            className="g-2",
        ),

        # dcc.Download component is responsible for handling the download
        dcc.Download(id="form-download-zones-csv"),

    ],
    fluid=True,
    )



@dash.callback(
    Output("form-name", "options"),  # Update the dropdown options for full name
    Input("form-status", "value"),  # Trigger this whenever form-status changes (or use other trigger)
)
def populate_name_dropdown(selected_value):
    try:
        # Retrieve token (authentication step)
        token = auth.get_token()
    except Exception as e:
        raise PreventUpdate  # If auth fails, don't update anything

    # Fetch names using your custom function
    # Replace 'SPORT_ORG_ENDPOINT' with your actual endpoint for fetching names (or profiles)
    filters = {'sport_org_id': 13}
    names = fetch_profiles(token, filters)  # Assuming this fetches profiles with full names and ids

    # Format the names for the dropdown
    rv = [
        {
            "label": f"{p['person']['first_name']} {p['person']['last_name']}",
            "value": int(p["id"]),   # <-- profile id
        }
        for p in names
    ]
    return rv
    


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
            "Type": mode,      # ✅ auto-populate
            "T_PO": None,
            "A_PO": None,
            "HR": None,
            "La": None,
            "V02": None,
            "rate": None,
            "split": None,     # computed
            "rpe": None,
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
    Output("form-avg-PO", "children"),
    Output("form-avg-HR", "children"),
    Output("form-avg-rate", "children"),
    Output("form-row-count", "children"),
    Input("form-items-table", "data"),
)
def update_summary_cards(rows):
    rows = rows or []

    po_vals = [to_float(r.get("A_PO")) for r in rows]
    hr_vals = [to_float(r.get("HR")) for r in rows]
    rate_vals = [to_float(r.get("rate")) for r in rows]

    po_vals = [v for v in po_vals if v is not None]
    hr_vals = [v for v in hr_vals if v is not None]
    rate_vals = [v for v in rate_vals if v is not None]

    po_avg = (sum(po_vals) / len(po_vals)) if po_vals else None
    hr_avg = (sum(hr_vals) / len(hr_vals)) if hr_vals else None
    rate_avg = (sum(rate_vals) / len(rate_vals)) if rate_vals else None

    po_txt = f"{po_avg:.1f} W" if po_avg is not None else "—"
    hr_txt = f"{hr_avg:.1f} bpm" if hr_avg is not None else "—"
    rate_txt = f"{rate_avg:.1f} spm" if rate_avg is not None else "—"

    return po_txt,hr_txt, rate_txt, str(len(rows))

@dash.callback(
    Output("form-items-table", "data", allow_duplicate=True),
    Input("form-items-table", "data_timestamp"),
    State("form-items-table", "data"),
    prevent_initial_call=True,
)

def compute_split_column(_, rows):
    rows = rows or []
    changed = False

    for r in rows:
        new_split = estimate_split_seconds(r.get("A_PO"))
        if r.get("split") != new_split:
            r["split"] = new_split
            changed = True

    return rows if changed else no_update



@dash.callback(
    Output("form-last-payload", "data"),
    Output("form-status-msg", "children"),
    Output("form-status-msg", "is_open"),
    Input("form-submit", "n_clicks"),
    Input("form-reset", "n_clicks"),
    State("form-name", "value"),      # will now be profile_id (int)
    State("form-mass", "value"),
    State("form-status", "value"),    # your Test Type (erg_C2, erg_RP3, row, etc.)
    State("form-mode", "value"),      # Max/Submax
    State("form-notes", "value"),
    State("form-items-table", "data"),
    prevent_initial_call=True,
)
def submit_or_reset(submit_clicks, reset_clicks, profile_id, mass, test_type, mode, notes, table_rows):
    trig = ctx.triggered_id

    if trig == "form-reset":
        payload = {"timestamp": datetime.now().isoformat(timespec="seconds"), "reset": True}
        return payload, "Form reset requested.", True

    if not submit_clicks:
        raise PreventUpdate

    # --- basic validation ---
    if profile_id is None:
        return no_update, "Please select an athlete before submitting.", True

    table_rows = table_rows or []
    if not isinstance(table_rows, list) or len(table_rows) == 0:
        return no_update, "No step data found. Add at least one row.", True

    # --- build a session id (lets you group steps later) ---
    session_ts = datetime.now().isoformat(timespec="seconds")
    session_id = f"{int(profile_id)}_{session_ts}"

    # --- clean/shape records: 1 record per step row ---
    records = []
    for r in table_rows:
        # skip fully empty rows (optional rule)
        has_any = any((r.get(k) not in (None, "", []) ) for k in ["T_PO", "A_PO", "HR", "La", "V02", "rate", "rpe"])
        if not has_any:
            continue

        records.append({
            "profile_id": int(profile_id),
            "session_id": session_id,
            "session_ts": session_ts,
            "body_mass_kg": mass,
            "test_type": test_type,   # erg_C2, erg_RP3, row, bike...
            "mode": mode,             # Max/Submax
            "notes": (notes or "").strip(),

            # step-level fields
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
        })

    if not records:
        return no_update, "All rows were empty — nothing to submit.", True

    payload = {
        "timestamp": session_ts,
        "form": {
            "profile_id": int(profile_id),
            "mass": mass,
            "test_type": test_type,
            "mode": mode,
            "notes": (notes or "").strip(),
        },
        "items": table_rows,
        "session_id": session_id,
        "records_preview_count": len(records),
    }

    # --- ingest into warehouse ---
    try:
        dataset, created = wc.ingest_raw(
            source_uuid=VO2_STEP_SOURCE_UUID,
            records=records,
            subject_field="profile_id",
            validate_client_side=False,
        )
        return payload, f"Submitted {created} row(s). Dataset UUID: {dataset['uuid']}", True

    except WarehouseClientError as e:
        return payload, f"Ingest failed: {e}", True


@dash.callback(
    Output("form-download-csv", "data"),  # This will trigger the CSV download
    Input("form-download-btn", "n_clicks"),
    State("form-items-table", "data"),  # Grab the table data to export as CSV
    prevent_initial_call=True
)
def download_csv(n_clicks, rows):
    if not n_clicks:
        raise PreventUpdate

    # Ensure rows is a list of dictionaries
    if not isinstance(rows, list):
        raise ValueError("Rows data should be a list of dictionaries")

    # Convert the rows (table data) to a pandas DataFrame
    try:
        df = pd.DataFrame(rows)
    except Exception as e:
        raise ValueError(f"Error converting rows to DataFrame: {e}")

    # Convert DataFrame to CSV (without index)
    csv_data = df.to_csv(index=False)

    # Return the data to be downloaded
    return dict(content=csv_data, filename="form_submission.csv", type="text/csv")
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

    # Ensure required columns exist even if table is empty
    for c in ["A_PO", "La", "HR"]:
        if c not in df.columns:
            df[c] = None

    # Coerce numeric
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
    fig_la.update_traces(marker=dict(color='blue'))

    fig_la = add_poly_fit(
        fig_la,
        df_la["A_PO"].values,
        df_la["La"].values,
        degree=2,
        name="La Power Fit",
        color = 'blue', 
        
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

    # --- Linear regression + line ---
    slope_txt = "—"
    intercept_txt = "—"

    df_hr_fit = df_hr.dropna(subset=["A_PO", "HR"])
    if len(df_hr_fit) >= 2:
        lr = sp.stats.linregress(df_hr_fit["A_PO"].values, df_hr_fit["HR"].values)
        slope = lr.slope
        intercept = lr.intercept

        slope_txt = f"{slope:.4f} bpm/W"
        intercept_txt = f"{intercept:.2f} bpm"

        x_fit = np.linspace(df_hr_fit["A_PO"].min(), df_hr_fit["A_PO"].max(), 100)
        y_fit = intercept + slope * x_fit

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
    State("form-max-hr", "value"),
)
def compute_zones(step_rows, max_hr_input):
    step_rows = step_rows or []
    df = pd.DataFrame(step_rows)

    if df.empty:
        return ZONES_DEFAULT_ROWS

    # Ensure numeric cols exist
    for c in ["HR", "La", "A_PO", "rate"]:
        if c not in df.columns:
            df[c] = None

    df["HR"] = pd.to_numeric(df["HR"], errors="coerce")
    df["La"] = pd.to_numeric(df["La"], errors="coerce")
    df["A_PO"] = pd.to_numeric(df["A_PO"], errors="coerce")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")

    # Need at least 2 La+HR points to interpolate lactate anchors
    df_la_hr = df.dropna(subset=["La", "HR"]).copy()
    if len(df_la_hr) < 2:
        return ZONES_DEFAULT_ROWS

    # Max HR for upper zones
    if max_hr_input is not None and max_hr_input != "":
        hr_max = float(max_hr_input)
    else:
        hr_max = df["HR"].max()
        if pd.isna(hr_max):
            hr_max = df_la_hr["HR"].max()

    hr_max = float(hr_max)

    # ---------- Interpolation helpers ----------
    # Interp HR at given lactate
    d_la_hr = df_la_hr.groupby("La", as_index=False)["HR"].mean().sort_values("La")
    la_vals = d_la_hr["La"].to_numpy(dtype=float)
    hr_vals = d_la_hr["HR"].to_numpy(dtype=float)

    def hr_at_la(target_la: float) -> float:
        # clamp to observed range
        if target_la <= la_vals.min():
            return float(hr_vals[0])
        if target_la >= la_vals.max():
            return float(hr_vals[-1])
        return float(np.interp(float(target_la), la_vals, hr_vals))

    # Interp PO at given lactate (if PO present)
    df_la_po = df.dropna(subset=["La", "A_PO"]).copy()
    po_at_la_ok = len(df_la_po) >= 2
    if po_at_la_ok:
        d_la_po = df_la_po.groupby("La", as_index=False)["A_PO"].mean().sort_values("La")
        la_po_vals = d_la_po["La"].to_numpy(dtype=float)
        po_vals = d_la_po["A_PO"].to_numpy(dtype=float)

        def po_at_la(target_la: float) -> float:
            if target_la <= la_po_vals.min():
                return float(po_vals[0])
            if target_la >= la_po_vals.max():
                return float(po_vals[-1])
            return float(np.interp(float(target_la), la_po_vals, po_vals))
    else:
        po_at_la = None  # type: ignore

    # Interp HR at a given PO (for PO-midpoint boundary)
    def hr_at_po(target_po: float) -> float | None:
        return _interp_y_at_x(df, "A_PO", "HR", target_po)

    # Interp PO at a given HR (for Z5/Z6 PO_low only)
    def po_at_hr(target_hr: float) -> float | None:
        return _interp_y_at_x(df, "HR", "A_PO", target_hr)

    def rate_at_hr(target_hr: float) -> float | None:
        return _interp_y_at_x(df, "HR", "rate", target_hr)

    def split_from_po(target_po: float | None) -> str | None:
        if target_po is None:
            return None
        sec = estimate_split_seconds(target_po)
        return format_split_mmss(sec)

    # Build a row with explicit HR and PO bounds
    def zone_row(
        zone_code: str,
        label: str,
        hr_low: float | None,
        hr_high: float | None,
        po_low: float | None,
        po_high: float | None,
        notes: str = "",
    ):
        # Rate bounds from HR bounds (if available)
        rate_low = rate_at_hr(hr_low) if hr_low is not None else None
        rate_high = rate_at_hr(hr_high) if hr_high is not None else None

        # Split bounds from PO bounds (if available)
        split_low = split_from_po(po_low)
        split_high = split_from_po(po_high)

        return {
            "Zone": f"{zone_code}/{label}",
            "HR_low": round(hr_low, 0) if hr_low is not None else None,
            "HR_high": round(hr_high, 0) if hr_high is not None else None,
            "PO_low": round(po_low, 1) if po_low is not None else None,
            "PO_high": round(po_high, 1) if po_high is not None else None,
            "Split_low": split_low,
            "Split_high": split_high,
            "Rate_low": round(rate_low, 1) if rate_low is not None else None,
            "Rate_high": round(rate_high, 1) if rate_high is not None else None,
            "Notes": notes or "",
        }

    # ---------- Physiology team zone definitions ----------
    LA_LOW_C6 = 1.5
    LA_2 = 2.0
    LA_4 = 4.0

    hr_15 = hr_at_la(LA_LOW_C6)
    hr_2 = hr_at_la(LA_2)
    hr_4 = hr_at_la(LA_4)

    # Ensure monotonic ordering in case of messy input
    hr_15, hr_2, hr_4 = sorted([hr_15, hr_2, hr_4])

    # Z1/C7: 100 bpm to low C6 (HR at 1.5 mmol)
    z1_lo = 100.0
    z1_hi = hr_15

    # Z2/C6: 1.5–2 mmol HR
    z2_lo = hr_15
    z2_hi = hr_2

    # Z3/C5 and Z4/C4 require PO@2 and PO@4, then midpoint PO
    if not po_at_la_ok:
        # Can't build PO-based zones without PO+La data
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

    # HR bounds for the PO boundaries
    hr_at_po2 = hr_at_po(po_lo)
    hr_at_pomid = hr_at_po(po_mid)
    hr_at_po4 = hr_at_po(po_hi)

    # If HR~PO interpolation fails (e.g., no overlap), fall back to lactate HR anchors
    if hr_at_po2 is None:
        hr_at_po2 = hr_2
    if hr_at_pomid is None:
        hr_at_pomid = (hr_2 + hr_4) / 2.0
    if hr_at_po4 is None:
        hr_at_po4 = hr_4

    # Z3/C5: 2 mmol W/HR to midpoint W
    z3_po_lo, z3_po_hi = po_lo, po_mid
    z3_hr_lo, z3_hr_hi = float(hr_at_po2), float(hr_at_pomid)

    # Z4/C4: midpoint W to 4 mmol W/HR
    z4_po_lo, z4_po_hi = po_mid, po_hi
    z4_hr_lo, z4_hr_hi = float(hr_at_pomid), float(hr_at_po4)

    # Z5/C3: 4 mmol HR to halfway to max HR (no need watts zone)
    z5_hr_lo = hr_4
    z5_hr_hi = (hr_4 + hr_max) / 2.0

    # Z6/C2/C1: C3 upper HR to max HR
    z6_hr_lo = z5_hr_hi
    z6_hr_hi = hr_max

    # PO lows for Z5/Z6 (only) from HR; PO_high intentionally None
    z5_po_low = po_at_hr(z5_hr_lo)
    z6_po_low = po_at_hr(z6_hr_lo)

    zones = [
        zone_row("Z1", "C7", z1_lo, z1_hi, po_at_hr(z1_lo), po_at_hr(z1_hi), notes="100 bpm → low C6 (HR@1.5)"),
        zone_row("Z2", "C6", z2_lo, z2_hi, po_at_hr(z2_lo), po_at_hr(z2_hi), notes="1.5–2 mmol HR"),
        zone_row("Z3", "C5", z3_hr_lo, z3_hr_hi, z3_po_lo, z3_po_hi, notes="2 mmol W → midpoint (2–4 mmol W)"),
        zone_row("Z4", "C4", z4_hr_lo, z4_hr_hi, z4_po_lo, z4_po_hi, notes="Midpoint → 4 mmol W"),
        zone_row("Z5", "C3", z5_hr_lo, z5_hr_hi, z5_po_low, None, notes="4 mmol HR → halfway to max HR"),
        zone_row("Z6", "C2/C1", z6_hr_lo, z6_hr_hi, z6_po_low, None, notes="Halfway → max HR"),
    ]

    return zones


def create_lactate_vs_po_plot(df):
    """
    Create a Plotly scatter plot for Lactate vs. Power Output (PO).
    """

    # Ensure necessary columns are present
    if not all(col in df.columns for col in ["La", "A_PO"]):
        raise ValueError("The dataframe must contain 'La' and 'A_PO' columns.")

    # Clean up the data by removing NaN values
    df_clean = df.dropna(subset=["La", "A_PO"])

    # Create a scatter plot of Lactate (La) vs Power Output (A_PO)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_clean["A_PO"],  # Power Output (W)
        y=df_clean["La"],     # Lactate (mmol/L)
        mode='markers',       # Scatter plot (points)
        marker=dict(color='blue', size=8, opacity=0.6),
        name="Lactate vs Power Output"
    ))

    # Add title and labels
    fig.update_layout(
        title="Lactate vs Power Output (PO)",
        xaxis_title="Power Output (W)",
        yaxis_title="Lactate (mmol/L)",
        template="plotly_white",  # Set a clean white theme for the plot
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig





@dash.callback(
    Output("form-download-zones-csv", "data"),  # Trigger the HR Zones CSV download
    Input("form-download-zones-btn", "n_clicks"),
    State("zones-table", "data"),  # Grab the HR zones table data
    prevent_initial_call=True
)
def download_zones_csv(n_clicks, rows):
    if not n_clicks:
        raise PreventUpdate

    # Ensure rows is a list of dictionaries (HR zones table data)
    if not isinstance(rows, list):
        raise ValueError("Rows data should be a list of dictionaries")

    # Convert the rows (zones data) to a pandas DataFrame
    try:
        df = pd.DataFrame(rows)
    except Exception as e:
        raise ValueError(f"Error converting rows to DataFrame: {e}")

    # Convert DataFrame to CSV (without index)
    csv_data = df.to_csv(index=False)

    # Return the data to be downloaded as a CSV
    return dict(content=csv_data, filename="HR_Training_Zones.csv", type="text/csv")