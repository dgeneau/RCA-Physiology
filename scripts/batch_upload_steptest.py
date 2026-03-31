# batch_upload_step_tests.py

from __future__ import annotations

from datetime import datetime
import math
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when running this script directly
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import fetch_profiles

from scripts.auth_cli import get_cli_token
from settings import SITE_URL, VO2_STEP_SOURCE_UUID
from warehouse import WarehouseAPIConfig, WarehouseClient, WarehouseClientError

# =========================================================
# CONFIG
# =========================================================
CSV_PATH = "/Users/danielgeneau/Library/CloudStorage/OneDrive-CanadianSportInstitutePacific/Rowing/Smartabase Exports/StepTestPull_20260129/rowing vo2 submax and max step testing excel report 1769720152478.csv"
SPORT_ORG_ID = 13
DRY_RUN = False        # set to False when ready to upload
CHUNK_SIZE = 500        # upload in chunks if needed

# Optional manual overrides if CSV athlete names do not exactly match platform names.
# Keys = names in CSV "About" column
# Values = profile_id (preferred) OR exact full name in your athlete system
MANUAL_PROFILE_MAP = {
    # "Athlete 1": 1234,
    # "Athlete 2": "Jane Smith",
}


# =========================================================
# CLIENT SETUP
# =========================================================
# Use CLI token flow (does not require a Flask request context)
cfg = WarehouseAPIConfig(base_url=SITE_URL)
wc = WarehouseClient(cfg, token_getter=get_cli_token)


# =========================================================
# HELPERS
# =========================================================
def clean_value(x):
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x = x.strip()
        return x if x != "" else None
    return x


def parse_rpe(x):
    """
    Converts strings like:
      '2- Fairly light' -> 2
      '7-Very hard'     -> 7
      '6'               -> 6
    """
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

    # fallback for messy historical labels
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


def build_profile_lookup():
    token = get_cli_token()
    profiles = fetch_profiles(token, {"sport_org_id": SPORT_ORG_ID})

    by_name = {}
    by_id = {}

    for p in profiles:
        pid = int(p["id"])
        full_name = f"{p['person']['first_name']} {p['person']['last_name']}".strip()
        by_name[full_name.lower()] = pid
        by_id[pid] = full_name

    return by_name, by_id


def resolve_profile_id(csv_name, by_name):
    """
    Resolve CSV athlete name to profile_id.

    Priority:
    1) direct numeric manual map
    2) manual map to exact platform full name
    3) exact name match in platform
    """
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


def prepare_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

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

    # clean strings
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(clean_value)

    # parse date dd-mm-yyyy
    df["test_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    if df["test_date"].isna().any():
        bad_rows = df[df["test_date"].isna()][["Date", "About"]]
        raise ValueError(f"Some dates could not be parsed:\n{bad_rows}")

    # numeric columns
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
    df["mode"] = df["Submax/Max"].apply(clean_value)  # same concept in your current schema
    df["notes_clean"] = df["Notes"].apply(clean_value)
    df["test_type_clean"] = df.apply(
        lambda r: map_test_type(r["Test Type"], r["Other Test Type"]),
        axis=1,
    )

    # Fill repeated session-level fields within athlete/date/test block
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


def build_records(df, by_name):
    df = df.copy()

    # resolve athlete -> profile_id
    df["profile_id"] = df["About"].apply(lambda x: resolve_profile_id(x, by_name))

    unmatched = sorted(df.loc[df["profile_id"].isna(), "About"].dropna().unique().tolist())
    if unmatched:
        raise ValueError(
            "Could not resolve these CSV athlete names to profile_id:\n"
            + "\n".join(f" - {name}" for name in unmatched)
            + "\n\nAdd them to MANUAL_PROFILE_MAP."
        )

    df["profile_id"] = df["profile_id"].astype(int)

    # build stable session grouping
    session_group = [
        "profile_id",
        "test_date",
        "test_type_clean",
        "notes_clean",
    ]

    # session index per athlete/date/type/notes block
    sessions = (
        df[session_group]
        .drop_duplicates()
        .sort_values(session_group)
        .reset_index(drop=True)
    )
    sessions["session_idx"] = range(1, len(sessions) + 1)

    df = df.merge(sessions, on=session_group, how="left")

    # create session_ts/session_id
    # using noon timestamp on test day + session_idx for uniqueness
    df["session_ts"] = df["test_date"].apply(
        lambda d: datetime(d.year, d.month, d.day, 12, 0, 0).isoformat(timespec="seconds")
    )
    df["session_id"] = df.apply(
        lambda r: f"{int(r['profile_id'])}_{r['test_date'].strftime('%Y%m%d')}_{int(r['session_idx']):03d}",
        axis=1,
    )

    records = []
    for _, r in df.iterrows():
        record = {
            "profile_id": int(r["profile_id"]),
            "session_id": r["session_id"],
            "session_ts": r["session_ts"],
            "test_date": r["test_date"].date().isoformat(),
            "body_mass_kg": clean_value(r["body_mass_kg"]),
            "test_type": clean_value(r["test_type_clean"]),
            "mode": clean_value(r["mode"]),
            "notes": clean_value(r["notes_clean"]),
            "step_no": clean_value(r["step_no"]),
            "step_type": clean_value(r["step_type"]),
            "target_po_w": clean_value(r["target_po_w"]),
            "actual_po_w": clean_value(r["actual_po_w"]),
            "hr_bpm": clean_value(r["hr_bpm"]),
            "lactate_mmol": clean_value(r["lactate_mmol"]),
            "vo2": clean_value(r["vo2"]),
            "rate_spm": clean_value(r["rate_spm"]),
            "split_sec_per_500": clean_value(r["split_sec_per_500"]),
            "rpe": clean_value(r["rpe"]),
            "time_s": None,  # not present in historical CSV
        }
        records.append(record)

    return records, df


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# =========================================================
# MAIN
# =========================================================
def main():
    if not VO2_STEP_SOURCE_UUID:
        raise ValueError("VO2_STEP_SOURCE_UUID is not set.")

    print("Loading profiles...")
    by_name, by_id = build_profile_lookup()

    print("Reading CSV...")
    df = prepare_dataframe(CSV_PATH)

    print("Transforming records...")
    records, df_final = build_records(df, by_name)

    print(f"Prepared {len(records)} records")
    print(f"Unique athletes: {df_final['profile_id'].nunique()}")
    print(f"Unique sessions: {df_final['session_id'].nunique()}")

    preview_cols = [
        "About", "profile_id", "test_date", "session_id",
        "step_no", "target_po_w", "actual_po_w", "hr_bpm", "lactate_mmol"
    ]
    print("\nPreview:")
    print(df_final[preview_cols].head(15).to_string(index=False))

    if DRY_RUN:
        print("\nDRY_RUN=True, no data uploaded.")
        return

    total_created = 0
    last_dataset = None

    print("\nUploading...")
    for i, batch in enumerate(chunked(records, CHUNK_SIZE), start=1):
        dataset, created = wc.ingest_raw(
            source_uuid=VO2_STEP_SOURCE_UUID,
            records=batch,
            subject_field="profile_id",
            validate_client_side=False,
        )
        last_dataset = dataset
        total_created += created
        print(f"Batch {i}: uploaded {created} records")

    print("\nDone.")
    print(f"Total uploaded: {total_created}")
    if last_dataset:
        print(f"Dataset UUID: {last_dataset['uuid']}")


if __name__ == "__main__":
    try:
        main()
    except WarehouseClientError as e:
        print(f"Warehouse error: {e}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise