#!/usr/bin/env python3
"""
update_daily.py

Fetches daily prices for all tickers present in the existing combined CSV (prefers nse_all_10y.csv)
and appends any missing daily rows. Saves a backup of the previous file before overwriting.

This script is safe to run daily (idempotent) and intended to be scheduled with cron using your venv python.
"""
import os
import sys
import shutil
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

# Configuration
POSSIBLE_FILES = ["/app/nse_all_10y.csv"]
BACKUP_DIR = "/app/backups"
BATCH_SIZE = 50


def find_data_file():
    for f in POSSIBLE_FILES:
        if os.path.exists(f):
            return f
    return None


def read_existing(file_path):
    """Try reading the existing CSV. Prefer reading MultiIndex header if present, else fall back."""
    try:
        df = pd.read_csv(file_path, index_col=0, header=[0,1], parse_dates=True, low_memory=False)
        print("Read existing file with MultiIndex columns")
        return df
    except Exception:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True, low_memory=False)
        print("Read existing file with flat columns")
        return df


def detect_base_symbols(columns):
    import re
    pattern = re.compile(r'(^.+?\.NS)(?:$|\.)')
    base = set()
    # Handle MultiIndex columns (common when data was saved with MultiIndex)
    try:
        # pandas MultiIndex has .nlevels
        if getattr(columns, 'nlevels', None) and columns.nlevels > 1:
            # take the first level (ticker names)
            first_level = [str(x) for x in columns.get_level_values(0)]
            for x in first_level:
                if x.endswith('.NS'):
                    base.add(x)
            return sorted(base)
    except Exception:
        pass

    # Fallback: columns is a flat Index of strings
    for c in columns:
        s = str(c)
        # Sometimes pandas flattens tuples into strings like "('RELIANCE.NS','Adj Close')"; try to extract
        # Try direct pattern match first
        m = pattern.match(s)
        if m:
            base.add(m.group(1))
            continue
        # Try to find something that looks like TICKER.NS inside the string
        m2 = re.search(r'([A-Z0-9\.\-]+\.NS)', s)
        if m2:
            base.add(m2.group(1))

    return sorted(base)


def backup_file(file_path):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dest = os.path.join(BACKUP_DIR, f"{os.path.basename(file_path)}.{timestamp}.bak")
    shutil.copy2(file_path, dest)
    print(f"Backup saved to {dest}")


def fetch_new_data(tickers, start_date):
    """Download daily data from start_date (inclusive) to today for the provided tickers in batches.
    Returns a DataFrame with MultiIndex columns like yfinance returns.
    """
    all_batches = []
    today = datetime.now().date()
    if start_date >= today:
        print("No new data to fetch (start_date >= today)")
        return None

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i+BATCH_SIZE]
        print(f"Fetching batch {i//BATCH_SIZE + 1} of {(len(tickers)+BATCH_SIZE-1)//BATCH_SIZE} (size={len(batch)}) from {start_date}")
        try:
            df_batch = yf.download(batch, start=start_date.strftime('%Y-%m-%d'), interval='1d', group_by='ticker', threads=True, ignore_tz=True, progress=False)
        except Exception as e:
            print(f"Batch download failed: {e}")
            continue
        if df_batch is None or df_batch.empty:
            print("No data returned for this batch")
            continue
        all_batches.append(df_batch)

    if not all_batches:
        return None

    new_data = pd.concat(all_batches, axis=1)
    return new_data


def merge_and_save(existing_df, new_df, out_file):
    # Align indexes: ensure datetime index
    if not isinstance(existing_df.index, pd.DatetimeIndex):
        existing_df.index = pd.to_datetime(existing_df.index)
    if not isinstance(new_df.index, pd.DatetimeIndex):
        new_df.index = pd.to_datetime(new_df.index)

    combined = pd.concat([existing_df, new_df])
    # keep rows unique by index (dates); keep the earliest (existing) rows first, so drop duplicates keeping first
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.sort_index()

    # Save backup and write
    backup_file(out_file)
    combined.to_csv(out_file)
    print(f"Updated data saved to {out_file}")


def main():
    data_file = find_data_file()
    if not data_file:
        print("No data file found. Run the downloader (6mo.py) first to create one.")
        return

    print(f"Using data file: {data_file}")
    existing = read_existing(data_file)

    # detect last date present using index
    if isinstance(existing.index, pd.DatetimeIndex):
        last_date = existing.index.max().date()
    else:
        # fallback: try to find last non-null date by scanning first column
        try:
            idx = pd.to_datetime(existing.iloc[:,0].dropna().index)
            last_date = idx.max().date()
        except Exception:
            last_date = None

    if last_date is None:
        print("Could not determine last date in existing data. Aborting.")
        return

    start_date = last_date + timedelta(days=1)
    print(f"Last date in file: {last_date}. Will fetch from: {start_date}")

    # detect tickers
    base_symbols = detect_base_symbols(existing.columns)
    if not base_symbols:
        print("No tickers detected in existing file. Aborting.")
        return

    # Fetch new data
    new_data = fetch_new_data(base_symbols, start_date)
    if new_data is None:
        print("No new data fetched. Nothing to update.")
        return

    # Merge and save to same file
    merge_and_save(existing, new_data, data_file)


if __name__ == '__main__':
    print(f"\n=== Update started at {datetime.now()} ===")
    try:
        main()
    except Exception as e:
        print(f"Error in main: {e}")
        sys.exit(1)
    print(f"=== Update completed at {datetime.now()} ===\n")
