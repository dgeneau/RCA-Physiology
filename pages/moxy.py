import base64
import io

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

dash.register_page(__name__, path="/moxy", name="Moxy Analysis")


# -----------------------------
# Helpers
# -----------------------------
def format_date(date_str: str) -> str:
    month_names = {
        "1": "January",
        "2": "February",
        "3": "March",
        "4": "April",
        "5": "May",
        "6": "June",
        "7": "July",
        "8": "August",
        "9": "September",
        "10": "October",
        "11": "November",
        "12": "December",
    }

    if not isinstance(date_str, str) or "-" not in date_str:
        return str(date_str)

    month, day = date_str.split("-")
    month_name = month_names.get(month.lstrip("0"), month)
    return f"{month_name}-{day}"


def parse_uploaded_csv(contents: str) -> pd.DataFrame:
    """
    Decode Dash upload contents and read the Moxy CSV.
    Matches the original Streamlit behavior: skip first 3 rows.
    """
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode("utf-8")), skiprows=3)


def parse_filename_parts(filename: str) -> dict:
    """
    Original Streamlit app assumed:
    <something>_<first>_<last>_<year...>
    """
    result = {"first": "", "last": "", "year": "", "display_name": filename or ""}

    if not filename:
        return result

    parts = filename.split("_")
    try:
        result["first"] = parts[1]
        result["last"] = parts[2]
        result["year"] = parts[3][:4]
        result["display_name"] = f"{result['first']} {result['last']} {result['year']}".strip()
    except Exception:
        # Fallback if filename format is different
        result["display_name"] = filename

    return result


def build_figure(
    dff: pd.DataFrame,
    data_sel: str,
    normalize: bool,
    title_text: str,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    sm_avg = dff["SmO2 Averaged"].copy()
    sm_live = dff["SmO2 Live"].copy()
    thb = dff["THb"].copy()

    # Match original behavior: use interior samples for max
    sm_avg_ref = sm_avg.iloc[20:-20] if len(sm_avg) > 40 else sm_avg
    sm_live_ref = sm_avg.iloc[20:-20] if len(sm_avg) > 40 else sm_avg
    thb_ref = thb.iloc[20:-20] if len(thb) > 40 else thb

    sm_avg_max = sm_avg_ref.max() if len(sm_avg_ref) else 1
    sm_live_max = sm_live_ref.max() if len(sm_live_ref) else 1
    thb_max = thb_ref.max() if len(thb_ref) else 1

    if normalize:
        if pd.notna(sm_avg_max) and sm_avg_max != 0:
            sm_avg = sm_avg / sm_avg_max
        if pd.notna(sm_live_max) and sm_live_max != 0:
            sm_live = sm_live / sm_live_max
        if pd.notna(thb_max) and thb_max != 0:
            thb = thb / thb_max

    if data_sel == "SmO2":
        fig.add_trace(
            go.Scatter(y=sm_avg, mode="lines", name="SmO2 Averaged"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(y=sm_live, mode="lines", name="SmO2 Live"),
            secondary_y=False,
        )

    elif data_sel == "THb":
        fig.add_trace(
            go.Scatter(y=thb, mode="lines", name="THb"),
            secondary_y=False,
        )

    else:  # All Measures
        fig.add_trace(
            go.Scatter(y=sm_avg, mode="lines", name="SmO2 Averaged"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(y=sm_live, mode="lines", name="SmO2 Live"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(y=thb, mode="lines", name="THb"),
            secondary_y=True,
        )

    fig.update_layout(
        title={"text": title_text, "font": {"size": 20}},
        xaxis_title="Sample Number",
        template="plotly_white",
        margin=dict(l=40, r=40, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=.8),
        height=600,
    )

    if data_sel == "All Measures":
        fig.update_yaxes(title_text="SmO2", secondary_y=False)
        fig.update_yaxes(title_text="THb", secondary_y=True)
    else:
        fig.update_yaxes(title_text=data_sel, secondary_y=False)

    return fig


# -----------------------------
# Layout
# -----------------------------
layout = dbc.Container(
    [
        dcc.Store(id="moxy-data-store"),
        dcc.Store(id="moxy-meta-store"),

        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H2("Rowing Canada Moxy Analysis", className="mb-3"),
                    ]
                ),
                width=12,
            ),
            className="mt-3",
        ),

        dbc.Row(
            dbc.Col(
                dcc.Upload(
                    id="moxy-upload",
                    children=dbc.Button("Select Moxy Data", color="primary"),
                    multiple=False,
                ),
                width="auto",
            ),
            className="mb-3",
        ),

        html.Div(id="moxy-upload-status", className="mb-3"),

        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Select Session"),
                                    dcc.Dropdown(
                                        id="moxy-session",
                                        placeholder="Select session",
                                        clearable=False,
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Select Data"),
                                    dcc.Dropdown(
                                        id="moxy-data-select",
                                        options=[
                                            {"label": "SmO2", "value": "SmO2"},
                                            {"label": "THb", "value": "THb"},
                                            {"label": "All Measures", "value": "All Measures"},
                                        ],
                                        value="SmO2",
                                        clearable=False,
                                    ),
                                ],
                                md=6,
                            ),
                        ],
                        className="mb-3",
                    ),

                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Checklist(
                                    options=[{"label": "Crop Data", "value": "crop"}],
                                    value=[],
                                    id="moxy-crop-toggle",
                                    switch=True,
                                ),
                                md=3,
                            ),
                            dbc.Col(
                                dbc.Checklist(
                                    options=[{"label": "Normalize Data to Max", "value": "norm"}],
                                    value=[],
                                    id="moxy-normalize-toggle",
                                    switch=True,
                                ),
                                md=4,
                            ),
                        ],
                        className="mb-3",
                    ),

                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Starting Index"),
                                    dbc.Input(
                                        id="moxy-start-index",
                                        type="number",
                                        value=0,
                                        step=1,
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Ending Index"),
                                    dbc.Input(
                                        id="moxy-end-index",
                                        type="number",
                                        value=-1,
                                        step=1,
                                    ),
                                ],
                                md=6,
                            ),
                        ],
                        id="moxy-crop-controls",
                        className="mb-3",
                        style={"display": "none"},
                    ),
                ]
            ),
            className="mb-4",
        ),

        dbc.Row(
            dbc.Col(
                dcc.Graph(id="moxy-graph"),
                width=12,
            )
        ),
    ],
    fluid=True,
)


# -----------------------------
# Callbacks
# -----------------------------
@callback(
    Output("moxy-data-store", "data"),
    Output("moxy-meta-store", "data"),
    Output("moxy-session", "options"),
    Output("moxy-session", "value"),
    Output("moxy-upload-status", "children"),
    Input("moxy-upload", "contents"),
    State("moxy-upload", "filename"),
    prevent_initial_call=True,
)
def load_file(contents, filename):
    if contents is None:
        return None, None, [], None, dbc.Alert("Upload Data", color="secondary")

    try:
        df = parse_uploaded_csv(contents)

        if "Session Ct" not in df.columns:
            return (
                None,
                None,
                [],
                None,
                dbc.Alert("Uploaded file does not contain 'Session Ct'.", color="danger"),
            )

        meta = parse_filename_parts(filename)
        sessions = sorted(df["Session Ct"].dropna().unique().tolist())
        session_options = [{"label": str(s), "value": s} for s in sessions]
        default_session = sessions[0] if sessions else None

        return (
            df.to_json(date_format="iso", orient="split"),
            meta,
            session_options,
            default_session,
            dbc.Alert(f"Loaded: {filename}", color="success"),
        )

    except Exception as e:
        return (
            None,
            None,
            [],
            None,
            dbc.Alert(f"Error loading file: {e}", color="danger"),
        )


@callback(
    Output("moxy-crop-controls", "style"),
    Input("moxy-crop-toggle", "value"),
)
def toggle_crop_controls(crop_value):
    if crop_value and "crop" in crop_value:
        return {"display": "flex"}
    return {"display": "none"}


@callback(
    Output("moxy-graph", "figure"),
    Input("moxy-data-store", "data"),
    Input("moxy-meta-store", "data"),
    Input("moxy-session", "value"),
    Input("moxy-data-select", "value"),
    Input("moxy-crop-toggle", "value"),
    Input("moxy-normalize-toggle", "value"),
    Input("moxy-start-index", "value"),
    Input("moxy-end-index", "value"),
)
def update_graph(
    data_json,
    meta,
    session,
    data_sel,
    crop_value,
    normalize_value,
    start_idx,
    end_idx,
):
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "Upload Data",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 18},
            }
        ],
        height=500,
    )

    if data_json is None or session is None:
        return empty_fig

    try:
        df = pd.read_json(io.StringIO(data_json), orient="split")
        dff = df.loc[df["Session Ct"] == session].reset_index(drop=True)

        if dff.empty:
            return empty_fig

        if crop_value and "crop" in crop_value:
            start_idx = 0 if start_idx is None else int(start_idx)
            end_idx = -1 if end_idx is None else int(end_idx)

            if end_idx == -1:
                dff = dff.iloc[start_idx:].reset_index(drop=True)
            else:
                dff = dff.iloc[start_idx:end_idx].reset_index(drop=True)

        date_val = ""
        if "mm-dd" in dff.columns and len(dff) > 0:
            date_val = format_date(str(dff["mm-dd"].iloc[0]))

        first = meta.get("first", "") if meta else ""
        last = meta.get("last", "") if meta else ""
        year = meta.get("year", "") if meta else ""

        title_text = f"Moxy Sensor Analysis {first} {last} {date_val} {year}".strip()

        normalize = bool(normalize_value and "norm" in normalize_value)

        return build_figure(
            dff=dff,
            data_sel=data_sel or "SmO2",
            normalize=normalize,
            title_text=title_text,
        )

    except Exception as e:
        error_fig = go.Figure()
        error_fig.update_layout(
            template="plotly_white",
            annotations=[
                {
                    "text": f"Error building figure: {e}",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 16},
                }
            ],
            height=500,
        )
        return error_fig