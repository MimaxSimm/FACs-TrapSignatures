#!/usr/bin/env python3
"""
Downloads the archived fit/MCMC results from Zenodo and unpacks them into
results/FitResults and results/MCMCResults.

Fill in ZENODO_RECORD_ID below once you've created and published (or
opened as a draft) the Zenodo record - it's the number in the record's
URL, e.g. for https://zenodo.org/records/1234567 the ID is 1234567.

Usage:
    python scripts/download_results.py
    python scripts/download_results.py --record-id 1234567
"""
import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

# EDIT ME after publishing to Zenodo:
ZENODO_RECORD_ID = "22791430"  # e.g. 1234567

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def fetch_record(record_id: str) -> dict:
    import json

    url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def download_and_extract(file_url: str, dest_dir: Path):
    print(f"  downloading {file_url} ...")
    with urllib.request.urlopen(file_url) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest_dir)
    print(f"  extracted into {dest_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-id", default=ZENODO_RECORD_ID)
    args = parser.parse_args()

    if not args.record_id:
        print(
            "No Zenodo record ID set. Either edit ZENODO_RECORD_ID at the top of "
            "this script, or pass --record-id 1234567.",
            file=sys.stderr,
        )
        sys.exit(1)

    record = fetch_record(args.record_id)
    files = {f["key"]: f["links"]["self"] for f in record.get("files", [])}

    if "FitResults.zip" in files:
        download_and_extract(files["FitResults.zip"], RESULTS_DIR)
    else:
        print("  ! FitResults.zip not found in this Zenodo record")

    if "MCMCResults.zip" in files:
        download_and_extract(files["MCMCResults.zip"], RESULTS_DIR)
    else:
        print("  ! MCMCResults.zip not found in this Zenodo record")

    print("Done.")


if __name__ == "__main__":
    main()
