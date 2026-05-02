"""
Prepare the small self-contained Unsupervised-Human-Motion-Analysis demo dataset.

This utility slices a few short excerpts from the full local workbook and joins
them with the existing sparse activity markers. The generated files contain
anonymised subject IDs, group labels, time, and every numeric sensor channel in
the selected sheets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEMO_SUBJECTS = [
    ("S1", "Control"),
    ("S2", "Control"),
    ("S13", "Injured"),
    ("S17", "Injured"),
]

METADATA_COLUMNS = {"Time", "time", "Marker", "Group", "sheet", "GroundTruth"}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(description="Create the small demo dataset from full local project outputs.")
    parser.add_argument(
        "--workbook",
        default=str(project_root / "data" / "data-multi.xlsx"),
        help="Full local sensor workbook.",
    )
    parser.add_argument(
        "--labels",
        default=str(project_root / "results_data-multi"),
        help="Full local results folder containing S*/step1_clean_results.csv for aligned labels.",
    )
    parser.add_argument(
        "--out",
        default=str(script_dir / "Data"),
        help="Output folder for demo CSV files.",
    )
    parser.add_argument("--seconds", type=float, default=20.0, help="Number of seconds to keep per subject.")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workbook_path = Path(args.workbook).resolve()
    labels_root = Path(args.labels).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not workbook_path.exists():
        raise FileNotFoundError(f"Missing source workbook: {workbook_path}")

    workbook = pd.ExcelFile(workbook_path)
    manifest_rows = []
    end_time = args.start + args.seconds

    for sheet, group in DEMO_SUBJECTS:
        label_path = labels_root / sheet / "step1_clean_results.csv"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing aligned label CSV for {sheet}: {label_path}")
        if sheet not in workbook.sheet_names:
            raise ValueError(f"Workbook does not contain sheet {sheet}.")

        raw = pd.read_excel(workbook, sheet_name=sheet)
        time_col = "time" if "time" in raw.columns else "Time"
        if time_col not in raw.columns:
            raise ValueError(f"{sheet} has no time column.")

        labels = pd.read_csv(label_path, usecols=["Time", "Marker", "Group"])
        raw_time = pd.to_numeric(raw[time_col], errors="coerce")
        keep_mask = (raw_time >= args.start) & (raw_time < end_time)
        raw_demo = raw.loc[keep_mask].reset_index(drop=True).copy()
        label_demo = labels.loc[keep_mask].reset_index(drop=True).copy()

        sensor_cols = []
        for col in raw_demo.columns:
            if col in METADATA_COLUMNS:
                continue
            numeric = pd.to_numeric(raw_demo[col], errors="coerce")
            if numeric.notna().any():
                raw_demo[col] = numeric
                sensor_cols.append(col)

        if not sensor_cols:
            raise RuntimeError(f"No numeric sensor columns found for {sheet}.")

        demo = pd.DataFrame(
            {
                "Time": pd.to_numeric(raw_demo[time_col], errors="coerce") - float(args.start),
                "Marker": label_demo["Marker"],
                "Group": group,
            }
        )
        for col in sensor_cols:
            demo[col] = raw_demo[col]

        if demo.empty:
            raise RuntimeError(f"No rows selected for {sheet}; check --start/--seconds.")

        out_name = f"{sheet}_demo_signals.csv"
        out_path = out_dir / out_name
        demo.to_csv(out_path, index=False)

        manifest_rows.append(
            {
                "sheet": sheet,
                "group": group,
                "file": out_name,
                "rows": int(len(demo)),
                "seconds": float(args.seconds),
                "sensor_columns": int(len(sensor_cols)),
            }
        )
        print(f"[OK] {sheet}: {len(demo)} rows, {len(sensor_cols)} sensor columns -> {out_path}")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "demo_subjects.csv", index=False)
    print(f"[OK] Manifest -> {out_dir / 'demo_subjects.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
