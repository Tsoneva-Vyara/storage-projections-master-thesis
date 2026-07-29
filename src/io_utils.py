"""
io_utils.py - dataset discovery and loading.

Finds the input Excel file in the current working directory, as it uses the
candidate list in config.CANDIDATE_FILES, loads it, and renames the source
columns to the internal names used elsewhere in the pipeline.
"""
import os
import sys
import pandas as pd

from config import CANDIDATE_FILES, RENAME


def discover_dataset_file():
    """
    Look for the dataset in the current working directory.

    Returns the first filename from ``config.CANDIDATE_FILES`` that exists
    on disk.  Prints an error listing all candidates and exits with code 1
    if none of them are found - the pipeline cannot proceed without data.
    """
    for f in CANDIDATE_FILES:
        if os.path.exists(f):
            return f
    print("\nERROR: dataset file not found.  Expected one of:")
    for c in CANDIDATE_FILES:
        print(f"  {c}")
    sys.exit(1)


def load_dataset(path):
    """
    Read the Excel file and apply the source-to-internal column renaming.

    Only columns, which are listed in ``config.RENAME`` are renamed; unmapped columns
    keep their original names and are simply ignored by downstream stages.
    All-NA columns are dropped.
    """
    print(f"\nLoading {path} ...")
    df = pd.read_excel(path, sheet_name=0)
    print(f"  raw load: {len(df)} rows × {len(df.columns)} cols")
    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})
    df = df.dropna(how="all", axis=1)
    return df