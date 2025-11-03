import pandas as pd
import numpy as np
import os
import argparse
from typing import List, Optional


COMMON_DATE_COLS = ["date", "Date", "report_date", "reported_on"]
COMMON_CASE_COLS = ["new_cases", "cases", "confirmed", "case_count", "total_cases", "cumulative_cases"]


def find_raw_csvs(raw_dir: str = "data/raw") -> List[str]:
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    csvs = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.lower().endswith('.csv')]
    return sorted(csvs)


def inspect_files(csv_paths: List[str], nrows: int = 3):
    info = {}
    for p in csv_paths:
        try:
            df = pd.read_csv(p, nrows=nrows)
        except Exception as e:
            print(f"Could not read {p}: {e}")
            continue
        info[p] = list(df.columns)
        print(f"File: {p}\n Columns: {info[p]}\n")
    return info


def detect_date_col(cols: List[str]) -> Optional[str]:
    for c in COMMON_DATE_COLS:
        if c in cols:
            return c
    # try fuzzy: any column containing 'date'
    for c in cols:
        if 'date' in c.lower():
            return c
    return None


def detect_cases_col(cols: List[str]) -> Optional[str]:
    for c in COMMON_CASE_COLS:
        if c in cols:
            return c
    # fuzzy: pick numeric columns with 'case' or 'confirmed' in name
    for c in cols:
        low = c.lower()
        if ('case' in low or 'confirm' in low) and True:
            return c
    return None


def build_timeseries(csv_paths: List[str], date_col: str, cases_col: str, out_path: str = "data/processed/cases_timeseries.csv") -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Skipping {p}, read error: {e}")
            continue
        if date_col not in df.columns or cases_col not in df.columns:
            print(f"Skipping {p}, required columns not present")
            continue
        # parse date
        df = df[[date_col, cases_col]].copy()
        df = df.rename(columns={date_col: 'date', cases_col: 'cases'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        # coerce cases to numeric
        df['cases'] = pd.to_numeric(df['cases'], errors='coerce').fillna(0)
        frames.append(df)

    if not frames:
        raise ValueError("No valid frames to build timeseries from")

    all_df = pd.concat(frames, axis=0)
    # aggregate by date
    ts = all_df.groupby('date', as_index=True)['cases'].sum().sort_index()
    ts = ts.rename('daily_cases').to_frame()
    ts['cumulative_cases'] = ts['daily_cases'].cumsum()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ts.to_csv(out_path)
    print(f"Saved timeseries to {out_path}")
    return ts


def main(auto: bool = False, date_col: Optional[str] = None, cases_col: Optional[str] = None):
    csvs = find_raw_csvs()
    if not csvs:
        print("No CSV files found in data/raw/. Run the ingestion script first.")
        return

    if not auto:
        print("Found CSVs:")
        for p in csvs:
            print(' -', p)
        print("\nInspecting columns (first few rows)...\n")
        info = inspect_files(csvs)
        # suggest candidates
        # choose first file to auto-detect from
        first_cols = info[csvs[0]] if csvs[0] in info else []
        suggested_date = detect_date_col(first_cols)
        suggested_cases = detect_cases_col(first_cols)
        print(f"Suggested date column: {suggested_date}")
        print(f"Suggested case column: {suggested_cases}")
        print("If suggestions are correct, re-run with --auto --date-col DATE --cases-col CASES or provide them now.")
        return

    # auto mode
    info = inspect_files(csvs)
    # try to find columns from first file that look appropriate
    for p, cols in info.items():
        if not date_col:
            date_col = detect_date_col(cols)
        if not cases_col:
            cases_col = detect_cases_col(cols)
        if date_col and cases_col:
            break

    if not date_col or not cases_col:
        raise ValueError(f"Could not auto-detect date/cases columns. Detected date: {date_col}, cases: {cases_col}")

    print(f"Using date_col={date_col}, cases_col={cases_col}")
    ts = build_timeseries(csvs, date_col, cases_col)
    print(ts.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Ghana COVID-19 raw CSVs into a timeseries')
    parser.add_argument('--auto', action='store_true', help='Auto-detect and build timeseries')
    parser.add_argument('--date-col', help='Date column name to use')
    parser.add_argument('--cases-col', help='Cases column name to use')
    args = parser.parse_args()
    main(auto=args.auto, date_col=args.date_col, cases_col=args.cases_col)
