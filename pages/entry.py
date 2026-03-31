
# pages/entry.py

from datetime import datetime, date
import pandas as pd
import numpy as np
import scipy as sp

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
import base64
import io

cfg = WarehouseAPIConfig(base_url=SITE_URL)
wc = WarehouseClient(cfg, token_getter=auth.get_token)

dash.register_page(__name__, path="/entry", name="Data Entry")

# === One-off batch upload settings (remove after use) ===
SPORT_ORG_ID = 13  # adjust to your org ID if needed
BATCH_UPLOAD_DRY_RUN = False  # set False to actually push data

# Set to False to hide the batch-upload UI; the batch code still remains available.
ENABLE_BATCH_UPLOAD_UI = True

# Optional overrides for athlete names in the uploaded CSV.
# Keys are values from the 'About' column; values are profile_id (preferred) or exact full name.
MANUAL_PROFILE_MAP = {
    #"Payton  Gauthier": 1234,
    # "Athlete 2": "Jane Smith",
}

# Athletes to ignore when uploading (their rows will be skipped).
# Use the exact string as it appears in the 'About' column of the CSV.
SKIP_ATHLETES = {
    "Annika Goodwyn", 
    "Cait Whittard", 
    "Carter Cranmer-Smith", 
    "Conor Dillon",
    "Janette Peachey",
    "Joseph McCoy", 
    "Kamal Elbogdadi", 
    "Lexi Shimnowski", 
    "Madelyn Vandermeer", 
    "Tess Friar", 
    "Daniel de Groot", 
    "Andrew Hubbard", 
    "Claire Ellison", 
    "Euan Coulson", 
    "Jack (John) Walkey",
    "Jennifer Casson", 
    "Joshua King", 
    "Kai Bartel",
    "Kyle Nummi", 
    "Leia Till",
    "Liam Keane",
    "Lisa Samuel",
    "Luke Gadsdon",
    "Michael Caryk",
    "Mitchell Rodgers",
    "Olivia McMurray", 
    "Rebecca Zimmerman",
    "William Simpson", 
    "Emily Munroe", 
    "Gavin Stone", 

}

# If set, only upload this athlete's rows (matching 'About', case-insensitive).
# Useful for testing a single athlete before uploading the full file.
TEST_ONLY_ATHLETE = None # e.g. "Jane Smith"

# =========================================================
# STEP TEST TABLE
# =========================================================
DEFAULT_ROWS = [
    {
        "step_no": 1,
        "Type": "Submax",
        "T_PO": None,
        "A_PO": None,
        "HR": None,
        "La": None,
        "V02": None,
        "rate": None,
        "split": None,
        "rpe": None,
        "time_s": None,
    },
    {
        "step_no": 2,
        "Type": "Submax",
        "T_PO": None,
        "A_PO": None,
        "HR": None,
        "La": None,
        "V02": None,
        "rate": None,
        "split": None,
        "rpe": None,
        "time_s": None,
    },
    {
        "step_no": 3,
        "Type": "Submax",
        "T_PO": None,
        "A_PO": None,
        "HR": None,
        "La": None,
        "V02": None,
        "rate": None,
        "split": None,
        "rpe": None,
        "time_s": None,
    },
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
    {"name": "Split time", "id": "split", "type": "numeric", "editable": False},
    {"name": "RPE", "id": "rpe", "type": "numeric"},
    {"name": "Time in Step (s)", "id": "time_s", "type": "numeric"},
]


# =========================================================
# ERG TEST TABLE (each row = one athlete)
# =========================================================
ERG_DEFAULT_ROWS = [
    {"row_no": 1, "profile_id": "", "distance_m": None, "stroke_rate_spm": None, "power_w": None, "time_s": None},
    {"row_no": 2, "profile_id": "", "distance_m": None, "stroke_rate_spm": None, "power_w": None, "time_s": None},
    {"row_no": 3, "profile_id": "", "distance_m": None, "stroke_rate_spm": None, "power_w": None, "time_s": None},
]

ERG_TABLE_COLUMNS = [
    {"name": "Row", "id": "row_no", "type": "numeric"},
    {"name": "Athlete", "id": "profile_id", "type": "text", "presentation": "dropdown"},
    {"name": "Distance (m)", "id": "distance_m", "type": "numeric", "presentation": "dropdown"},
    {"name": "Stroke Rate (spm)", "id": "stroke_rate_spm", "type": "numeric"},
    {"name": "Power (W)", "id": "power_w", "type": "numeric"},
    {"name": "Time (s)", "id": "time_s", "type": "numeric"},
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


# --------------------------------------------------
# ONE-OFF BATCH-UPLOAD HELPERS (remove after use)
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


def resolve_profile_id(csv_name, by_name):
    """Resolve CSV athlete name to profile_id."""
    csv_name = clean_value(csv_name)
    if csv_name is None:
        return None

    # manual direct mapping
    if csv_name in MANUAL_PROFILE_MAP:
        mapped = MANUAL_PROFILE_MAP[csv_name]
        if isinstance(mapped, int):
            return mapped
        if isinstance(mapped, str):
            return by_name.get(mapped.strip().lower())

    # exact match
    return by_name.get(str(csv_name).strip().lower())


def prepare_batch_dataframe(contents, filename):
    """Parse uploaded CSV contents into a cleaned DataFrame."""
    if not contents:
        raise ValueError("No file content provided.")

    header, encoded = contents.split(",", 1)
    data = base64.b64decode(encoded)

    if filename.lower().endswith(".csv"):
        df = pd.read_csv(io.StringIO(data.decode("utf-8")))
    else:
        raise ValueError("Only CSV uploads are supported.")

    required_cols = [
        "Date",
        "About",
        "Body Weight (kg)",
        "Test Type",
        "Other Test Type",
        "Notes",
        "Step Number",
        "Submax/Max",
        "Target PO",
        "Actual PO",
        "Heart Rate",
        "Blood Lactate",
        "VO2",
        "Stroke Rate",
        "Split",
        "RPE",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(clean_value)

    df["test_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    if df["test_date"].isna().any():
        bad_rows = df[df["test_date"].isna()][["Date", "About"]]
        raise ValueError(f"Some dates could not be parsed:\n{bad_rows}")

    numeric_map = {
        "Body Weight (kg)": "body_mass_kg",
        "Step Number": "step_no",
        "Target PO": "target_po_w",
        "Actual PO": "actual_po_w",
        "Heart Rate": "hr_bpm",
        "Blood Lactate": "lactate_mmol",
        "VO2": "vo2",
        "Stroke Rate": "rate_spm",
        "Split": "split_sec_per_500",
    }
    for src, dst in numeric_map.items():
        df[dst] = pd.to_numeric(df[src], errors="coerce")

    df["rpe"] = df["RPE"].apply(parse_rpe)
    df["step_type"] = df["Submax/Max"].apply(clean_value)
    df["mode"] = df["Submax/Max"].apply(clean_value)
    df["notes_clean"] = df["Notes"].apply(clean_value)
    df["test_type_clean"] = df.apply(
        lambda r: map_test_type(r["Test Type"], r["Other Test Type"]),
        axis=1,
    )

    session_group = [
        "About",
        "test_date",
        "Test Type",
        "Other Test Type",
        "Notes",
    ]

    df["body_mass_kg"] = (
        df.groupby(session_group, dropna=False)["body_mass_kg"]
        .transform(lambda s: s.ffill().bfill())
    )

    return df


def build_batch_records(df, by_name):
    """Turn a cleaned batch DataFrame into warehouse records.

    Returns:
        (records, skipped_names)
    """
    df = df.copy()
    df["profile_id"] = df["About"].apply(lambda x: resolve_profile_id(x, by_name))

    # Skip athletes explicitly listed in SKIP_ATHLETES
    skipped = sorted(
        df.loc[
            df["About"].isin(SKIP_ATHLETES),
            "About",
        ]
        .dropna()
        .unique()
        .tolist()
    )
    if skipped:
        df = df[~df["About"].isin(SKIP_ATHLETES)]

    # Report any remaining unmatched athletes
    unmatched = sorted(df.loc[df["profile_id"].isna(), "About"].dropna().unique().tolist())
    if unmatched:
        raise ValueError(
            "Could not resolve these CSV athlete names to profile_id:\n"
            + "\n".join(f" - {name}" for name in unmatched)
            + "\n\nAdd them to MANUAL_PROFILE_MAP, or add them to SKIP_ATHLETES to ignore them."
        )

    # Convert to int once we know all are resolved
    df["profile_id"] = df["profile_id"].astype(int)

    # Validate required numeric fields before ingesting, so we can give a helpful error
    missing_step_no = df[df["step_no"].isna()]
    if not missing_step_no.empty:
        first = missing_step_no.iloc[0]
        raise ValueError(
            "Missing 'Step Number' (required) for some rows. "
            f"First missing row: About={first.get('About')!r}, Date={first.get('Date')!r}. "
            "Please ensure all rows have a 'Step Number'."
        )

    session_group = [
        "profile_id",
        "test_date",
        "test_type_clean",
        "notes_clean",
    ]

    sessions = (
        df[session_group]
        .drop_duplicates()
        .sort_values(session_group)
        .reset_index(drop=True)
    )
    sessions["session_idx"] = range(1, len(sessions) + 1)

    df = df.merge(sessions, on=session_group, how="left")

    df["session_ts"] = df["test_date"].apply(
        lambda d: datetime(d.year, d.month, d.day, 12, 0, 0).isoformat(timespec="seconds")
    )
    df["session_id"] = df.apply(
        lambda r: f"{int(r['profile_id'])}_{r['test_date'].strftime('%Y%m%d')}_{int(r['session_idx']):03d}",
        axis=1,
    )

    records = []
    for _, r in df.iterrows():
        step_no_val = clean_value(r["step_no"])
        if step_no_val is None:
            # Should not happen due to earlier validation, but guard defensively.
            raise ValueError(
                f"Missing step number for athlete={r.get('About')!r}, date={r.get('Date')!r}."
            )
        try:
            step_no_val = int(float(step_no_val))
        except Exception:
            raise ValueError(
                f"Invalid step number {step_no_val!r} for athlete={r.get('About')!r}, date={r.get('Date')!r}."
            )

        records.append(
            {
                "profile_id": int(r["profile_id"]),
                "session_id": r["session_id"],
                "session_ts": r["session_ts"],
                "test_date": r["test_date"].date().isoformat(),
                "body_mass_kg": clean_value(r["body_mass_kg"]),
                "test_type": clean_value(r["test_type_clean"]),
                "mode": clean_value(r["mode"]),
                "notes": clean_value(r["notes_clean"]) or "",  # schema requires a string (not null)
                "step_no": step_no_val,
                "step_type": clean_value(r["step_type"]),
                "target_po_w": clean_value(r["target_po_w"]),
                "actual_po_w": clean_value(r["actual_po_w"]),
                "hr_bpm": clean_value(r["hr_bpm"]),
                "lactate_mmol": clean_value(r["lactate_mmol"]),
                "vo2": clean_value(r["vo2"]),
                "rate_spm": clean_value(r["rate_spm"]),
                "split_sec_per_500": clean_value(r["split_sec_per_500"]),
                "rpe": clean_value(r["rpe"]),
                "time_s": None,
            }
        )

    return records, skipped


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# END ONE-OFF BATCH-UPLOAD HELPERS


def estimate_split_seconds(power_w):
    p = to_float(power_w)
    if p is None or p <= 0:
        return None
    pace = 500.0 * ((2.8 / p) ** (1.0 / 3.0))
    return round(pace, 2)


def add_poly_fit(fig, x, y, degree=2, name="Fit", color="red"):
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
            line=dict(dash="solid", color=color),
        )
    )
    return fig


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
# LAYOUT
# =========================================================

# Toggle whether the one-off batch upload UI is shown.
# The batch upload logic remains available if you want to call it programmatically.
BATCH_UPLOAD_UI = (
    [
        html.H5("One-off Batch CSV Upload", className="mt-3"),
        dcc.Upload(
            id="batch-upload",
            children=html.Div(
                [
                    "Drag and drop a CSV file here, or click to select.",
                ]
            ),
            style={
                "width": "100%",
                "height": "80px",
                "lineHeight": "80px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "marginBottom": "8px",
            },
            accept=".csv",
            multiple=False,
        ),
        dbc.Button(
            "Upload CSV to Warehouse",
            id="batch-upload-submit",
            color="secondary",
            className="w-100 mb-2",
        ),
        dbc.Alert(id="batch-upload-status", color="info", is_open=False),
        html.Small(
            "(Remove this batch-upload section after you’ve completed the one-off import)",
            className="text-muted",
        ),
    ]
    if ENABLE_BATCH_UPLOAD_UI
    else []
)

layout = dbc.Container(
    [
        html.H2("Entry"),
        html.Div("Step test workflow and batch erg entry."),
        html.Hr(),
        dcc.Store(id="athlete-options-store"),

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
                                        "Entry Fields",
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
                                                                min=0,
                                                                step=0.1,
                                                                value=None,
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Test Date"),
                                                            dcc.DatePickerSingle(
                                                                id="form-test-date",
                                                                date=date.today().isoformat(),
                                                                display_format="YYYY-MM-DD",
                                                                first_day_of_week=1,
                                                                clearable=False,
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
                                                        md=12,
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
                                                        md=12,
                                                    ),
                                                ],
                                                className="g-3",
                                            ),
                                            html.Br(),
                                            dbc.Label("Notes"),
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

                                            *BATCH_UPLOAD_UI,
                                            html.Hr(),
                                            dbc.Alert(id="form-status-msg", color="success", is_open=False),
                                        ],
                                    ),
                                    md=4,
                                ),

                                dbc.Col(
                                    make_card(
                                        "Step Test Table",
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
                                                    dbc.Col(make_card("Average PO", html.H4(id="form-avg-PO", className="m-0")), md=3),
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
                                html.Div(
                                    [
                                        dbc.Button("Add row", id="erg-add-row", color="success", size="sm", className="me-2"),
                                        dbc.Button("Delete selected", id="erg-delete-rows", color="danger", size="sm", outline=True),
                                        dbc.Button("Download CSV", id="erg-download-btn", color="info", size="sm", outline=True, className="ms-2"),
                                    ],
                                    className="mb-2",
                                ),

                                html.Div(
                                    dash_table.DataTable(
                                        id="erg-items-table",
                                        data=ERG_DEFAULT_ROWS,
                                        columns=ERG_TABLE_COLUMNS,
                                        dropdown={
                                            "profile_id": {
                                                "clearable": True,
                                                "options": [],
                                            },
                                            "distance_m": {
                                                "clearable": False,
                                                "options": [
                                                    {"label": "2000", "value": 2000},
                                                    {"label": "6000", "value": 6000},
                                                ],
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
                                            "zIndex": 10,
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
                                        style_cell_conditional=[
                                            {"if": {"column_id": "row_no"}, "width": "70px"},
                                            {"if": {"column_id": "profile_id"}, "width": "260px", "textAlign": "left"},
                                            {"if": {"column_id": "distance_m"}, "width": "140px"},
                                            {"if": {"column_id": "stroke_rate_spm"}, "width": "150px"},
                                            {"if": {"column_id": "power_w"}, "width": "130px"},
                                            {"if": {"column_id": "time_s"}, "width": "130px"},
                                        ],
                                        css=[
                                            {
                                                "selector": ".dash-spreadsheet-container .Select-menu-outer",
                                                "rule": "display: block !important; z-index: 3000 !important; max-height: 260px;",
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container .Select-option",
                                                "rule": "color: #212529 !important; background-color: white !important; padding: 8px 10px !important;",
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container .Select-option.is-focused",
                                                "rule": "background-color: #f8f9fa !important;",
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
                                                """,
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container .Select-arrow-zone",
                                                "rule": """
                                                    padding-right: 6px !important;
                                                """,
                                            },
                                            {
                                                "selector": ".dash-spreadsheet-container .is-focused:not(.is-open) > .Select-control",
                                                "rule": """
                                                    background-color: #f8fbff !important;
                                                    box-shadow: inset 0 0 0 1px #86b7fe !important;
                                                """,
                                            },
                                        ],
                                    ),
                                    style={
                                        "position": "relative",
                                        "zIndex": 20,
                                        "marginBottom": "1rem",
                                    },
                                ),

                                dcc.Download(id="erg-download-csv"),

                                dbc.Row(
                                    [
                                        dbc.Col(make_card("Rows", html.H4(id="erg-row-count", className="m-0")), md=3),
                                        dbc.Col(make_card("Avg Power", html.H4(id="erg-avg-power", className="m-0")), md=3),
                                        dbc.Col(make_card("Avg Rate", html.H4(id="erg-avg-rate", className="m-0")), md=3),
                                        dbc.Col(make_card("Total Time", html.H4(id="erg-total-time", className="m-0")), md=3),
                                    ],
                                    className="g-2 mt-4",
                                ),
                            ],
                        )
                    ],
                )
            ],
            id="entry-tabs",
            active_tab="tab-step-test",
        ),
    ],
    fluid=True,
)


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

    filters = {"sport_org_id": 13}
    names = fetch_profiles(token, filters)

    return [
        {
            "label": f"{p['person']['first_name']} {p['person']['last_name']}",
            "value": int(p["id"]),
        }
        for p in names
    ]


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
            "distance_m": {
                "clearable": False,
                "options": [
                    {"label": "2000", "value": 2000},
                    {"label": "6000", "value": 6000},
                ],
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
        "distance_m": {
            "clearable": False,
            "options": [
                {"label": "2000", "value": 2000},
                {"label": "6000", "value": 6000},
            ],
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
    Output("form-avg-PO", "children"),
    Output("form-avg-HR", "children"),
    Output("form-avg-rate", "children"),
    Output("form-row-count", "children"),
    Input("form-items-table", "data"),
)
def update_summary_cards(rows):
    rows = rows or []

    po_vals = [to_float(r.get("A_PO")) for r in rows if to_float(r.get("A_PO")) is not None]
    hr_vals = [to_float(r.get("HR")) for r in rows if to_float(r.get("HR")) is not None]
    rate_vals = [to_float(r.get("rate")) for r in rows if to_float(r.get("rate")) is not None]

    po_avg = (sum(po_vals) / len(po_vals)) if po_vals else None
    hr_avg = (sum(hr_vals) / len(hr_vals)) if hr_vals else None
    rate_avg = (sum(rate_vals) / len(rate_vals)) if rate_vals else None

    po_txt = f"{po_avg:.1f} W" if po_avg is not None else "—"
    hr_txt = f"{hr_avg:.1f} bpm" if hr_avg is not None else "—"
    rate_txt = f"{rate_avg:.1f} spm" if rate_avg is not None else "—"

    return po_txt, hr_txt, rate_txt, str(len(rows))


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
    State("form-name", "value"),
    State("form-mass", "value"),
    State("form-test-date", "date"),
    State("form-status", "value"),
    State("form-mode", "value"),
    State("form-notes", "value"),
    State("form-items-table", "data"),
    prevent_initial_call=True,
)
def submit_or_reset(submit_clicks, reset_clicks, profile_id, mass, test_date, test_type, mode, notes, table_rows):
    trig = ctx.triggered_id

    if trig == "form-reset":
        payload = {"timestamp": datetime.now().isoformat(timespec="seconds"), "reset": True}
        return payload, "Form reset requested.", True

    if not submit_clicks:
        raise PreventUpdate

    if profile_id is None:
        return no_update, "Please select an athlete before submitting.", True

    if test_date is None:
        return no_update, "Please select a test date before submitting.", True

    table_rows = table_rows or []
    if not isinstance(table_rows, list) or len(table_rows) == 0:
        return no_update, "No step data found. Add at least one row.", True

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
        })

    if not records:
        return no_update, "All rows were empty — nothing to submit.", True

    payload = {
        "timestamp": session_ts,
        "form": {
            "profile_id": int(profile_id),
            "mass": mass,
            "test_date": test_date,
            "test_type": test_type,
            "mode": mode,
            "notes": (notes or "").strip(),
        },
        "items": table_rows,
        "session_id": session_id,
        "records_preview_count": len(records),
    }

    try:
        if not VO2_STEP_SOURCE_UUID:
            return payload, "Ingest failed: VO2_STEP_SOURCE_UUID is not set.", True

        dataset, created = wc.ingest_raw(
            source_uuid=VO2_STEP_SOURCE_UUID,
            records=records,
            subject_field="profile_id",
            validate_client_side=False,
        )
        return payload, f"Submitted {created} row(s). Dataset UUID: {dataset['uuid']}", True

    except WarehouseClientError as e:
        return payload, f"Ingest failed: {e}", True


# --------------------------------------------------
# One-off batch CSV upload callback (remove after use)
# --------------------------------------------------
if ENABLE_BATCH_UPLOAD_UI:

    @dash.callback(
        Output("batch-upload-status", "children"),
        Output("batch-upload-status", "color"),
        Output("batch-upload-status", "is_open"),
        Input("batch-upload-submit", "n_clicks"),
        State("batch-upload", "contents"),
        State("batch-upload", "filename"),
        prevent_initial_call=True,
    )
    def batch_upload(n_clicks, contents, filename):
        if not contents:
            return "No file uploaded.", "warning", True

        if not VO2_STEP_SOURCE_UUID:
            return (
                "Batch upload failed: VO2_STEP_SOURCE_UUID is not set. Set it in settings.py.",
                "danger",
                True,
            )

        try:
            df = prepare_batch_dataframe(contents, filename)

            if TEST_ONLY_ATHLETE:
                df = df[df["About"].str.strip().str.lower() == TEST_ONLY_ATHLETE.strip().lower()]
                if df.empty:
                    return (
                        f"No rows found for TEST_ONLY_ATHLETE='{TEST_ONLY_ATHLETE}'.", "warning", True
                    )

            token = auth.get_token()
            profiles = fetch_profiles(token, {"sport_org_id": SPORT_ORG_ID})
            by_name = {
                f"{p['person']['first_name']} {p['person']['last_name']}".strip().lower(): int(p['id'])
                for p in profiles
                if p.get('person')
            }

            records, skipped = build_batch_records(df, by_name)
            if BATCH_UPLOAD_DRY_RUN:
                msg = f"Dry run: prepared {len(records)} records (no upload)."
                if skipped:
                    msg += f" Skipped {len(skipped)} athletes: {', '.join(skipped)}."
                msg += " Change BATCH_UPLOAD_DRY_RUN to False to ingest."
                return msg, "info", True

            dataset, created = wc.ingest_raw(
                source_uuid=VO2_STEP_SOURCE_UUID,
                records=records,
                subject_field="profile_id",
                validate_client_side=False,
            )

            return (
                f"Uploaded {created} records (dataset {dataset.get('uuid')}).",
                "success",
                True,
            )

        except Exception as e:
            return f"Batch upload failed: {e}", "danger", True


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

    if len(df_la) >= 3:
        fig_la = add_poly_fit(
            fig_la,
            df_la["A_PO"].values,
            df_la["La"].values,
            degree=2,
            name="La Power Fit",
            color="blue",
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
    State("form-max-hr", "value"),
)
def compute_zones(step_rows, max_hr_input):
    step_rows = step_rows or []
    df = pd.DataFrame(step_rows)

    if df.empty:
        return ZONES_DEFAULT_ROWS

    for c in ["HR", "La", "A_PO", "rate"]:
        if c not in df.columns:
            df[c] = None

    df["HR"] = pd.to_numeric(df["HR"], errors="coerce")
    df["La"] = pd.to_numeric(df["La"], errors="coerce")
    df["A_PO"] = pd.to_numeric(df["A_PO"], errors="coerce")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")

    df_la_hr = df.dropna(subset=["La", "HR"]).copy()
    if len(df_la_hr) < 2:
        return ZONES_DEFAULT_ROWS

    if max_hr_input is not None and max_hr_input != "":
        hr_max = float(max_hr_input)
    else:
        hr_max = df["HR"].max()
        if pd.isna(hr_max):
            hr_max = df_la_hr["HR"].max()

    hr_max = float(hr_max)

    d_la_hr = df_la_hr.groupby("La", as_index=False)["HR"].mean().sort_values("La")
    la_vals = d_la_hr["La"].to_numpy(dtype=float)
    hr_vals = d_la_hr["HR"].to_numpy(dtype=float)

    def hr_at_la(target_la):
        if target_la <= la_vals.min():
            return float(hr_vals[0])
        if target_la >= la_vals.max():
            return float(hr_vals[-1])
        return float(np.interp(float(target_la), la_vals, hr_vals))

    df_la_po = df.dropna(subset=["La", "A_PO"]).copy()
    po_at_la_ok = len(df_la_po) >= 2
    if po_at_la_ok:
        d_la_po = df_la_po.groupby("La", as_index=False)["A_PO"].mean().sort_values("La")
        la_po_vals = d_la_po["La"].to_numpy(dtype=float)
        po_vals = d_la_po["A_PO"].to_numpy(dtype=float)

        def po_at_la(target_la):
            if target_la <= la_po_vals.min():
                return float(po_vals[0])
            if target_la >= la_po_vals.max():
                return float(po_vals[-1])
            return float(np.interp(float(target_la), la_po_vals, po_vals))
    else:
        po_at_la = None  # noqa

    def hr_at_po(target_po):
        return _interp_y_at_x(df, "A_PO", "HR", target_po)

    def po_at_hr(target_hr):
        return _interp_y_at_x(df, "HR", "A_PO", target_hr)

    def rate_at_hr(target_hr):
        return _interp_y_at_x(df, "HR", "rate", target_hr)

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
        zone_row("Z1", "C7", z1_lo, z1_hi, po_at_hr(z1_lo), po_at_hr(z1_hi), notes="100 bpm → low C6 (HR@1.5)"),
        zone_row("Z2", "C6", z2_lo, z2_hi, po_at_hr(z2_lo), po_at_hr(z2_hi), notes="1.5–2 mmol HR"),
        zone_row("Z3", "C5", z3_hr_lo, z3_hr_hi, z3_po_lo, z3_po_hi, notes="2 mmol W → midpoint (2–4 mmol W)"),
        zone_row("Z4", "C4", z4_hr_lo, z4_hr_hi, z4_po_lo, z4_po_hi, notes="Midpoint → 4 mmol W"),
        zone_row("Z5", "C3", z5_hr_lo, z5_hr_hi, z5_po_low, None, notes="4 mmol HR → halfway to max HR"),
        zone_row("Z6", "C2/C1", z6_hr_lo, z6_hr_hi, z6_po_low, None, notes="Halfway → max HR"),
    ]


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
    Input("erg-add-row", "n_clicks"),
    Input("erg-delete-rows", "n_clicks"),
    State("erg-items-table", "data"),
    State("erg-items-table", "selected_rows"),
    prevent_initial_call=True,
)
def modify_erg_table(add_clicks, del_clicks, rows, selected_rows):
    rows = rows or []
    selected_rows = selected_rows or []

    if ctx.triggered_id == "erg-add-row":
        next_no = (max([r.get("row_no") or 0 for r in rows]) + 1) if rows else 1
        rows.append({
            "row_no": next_no,
            "profile_id": "",
            "distance_m": None,
            "stroke_rate_spm": None,
            "power_w": None,
            "time_s": None,
        })
        return rows, []

    if ctx.triggered_id == "erg-delete-rows":
        if not selected_rows:
            return no_update, no_update
        keep = [r for i, r in enumerate(rows) if i not in set(selected_rows)]
        return keep, []

    return no_update, no_update


@dash.callback(
    Output("erg-row-count", "children"),
    Output("erg-avg-power", "children"),
    Output("erg-avg-rate", "children"),
    Output("erg-total-time", "children"),
    Input("erg-items-table", "data"),
)
def update_erg_summary(rows):
    rows = rows or []
    df = pd.DataFrame(rows)

    if df.empty:
        return "0", "—", "—", "—"

    for c in ["power_w", "stroke_rate_spm", "time_s"]:
        if c not in df.columns:
            df[c] = None

    p = pd.to_numeric(df["power_w"], errors="coerce")
    r = pd.to_numeric(df["stroke_rate_spm"], errors="coerce")
    t = pd.to_numeric(df["time_s"], errors="coerce")

    avg_p = p.mean(skipna=True)
    avg_r = r.mean(skipna=True)
    total_t = t.sum(skipna=True)

    avg_p_txt = f"{avg_p:.1f} W" if pd.notna(avg_p) else "—"
    avg_r_txt = f"{avg_r:.1f} spm" if pd.notna(avg_r) else "—"
    total_t_txt = f"{total_t:.0f} s" if pd.notna(total_t) else "—"

    return str(len(rows)), avg_p_txt, avg_r_txt, total_t_txt


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