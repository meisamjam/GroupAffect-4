"""
Download the GroupAffect-4 dataset from Zenodo.

Usage
-----
    python download_dataset.py --out-dir /path/to/data
    python download_dataset.py --record-id 1234567 --out-dir /path/to/data --token <personal-access-token>

The script fetches the Zenodo record, lists all files, downloads them into
<out-dir> preserving the original path structure, and verifies MD5 checksums.

A Zenodo personal access token is only required for records that are not yet
publicly published (e.g. during review).  For public records omit --token.
"""

import argparse
import hashlib
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit(
        "requests is required — install it with:  pip install requests\n"
        "or install all dependencies:             pip install -r requirements.txt"
    )

# ---------------------------------------------------------------------------
# Replace with the actual Zenodo record ID once the dataset is staged.
# The record ID is the number at the end of the Zenodo URL:
#   https://zenodo.org/records/<RECORD_ID>
# ---------------------------------------------------------------------------
DEFAULT_RECORD_ID = "ZENODO_RECORD_ID_PLACEHOLDER"

ZENODO_API = "https://zenodo.org/api/records/{record_id}"


def fetch_record(record_id: str, token: str | None) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = ZENODO_API.format(record_id=record_id)
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 404:
        sys.exit(f"Record {record_id} not found on Zenodo. Check the record ID.")
    if resp.status_code == 401:
        sys.exit("Access denied. Pass a valid --token for restricted records.")
    resp.raise_for_status()
    return resp.json()


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while data := fh.read(chunk):
            h.update(data)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_md5: str | None, token: str | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {dest.name}: {pct:5.1f}%", end="", flush=True)
    print()

    if expected_md5:
        actual = md5(dest)
        if actual != expected_md5:
            sys.exit(
                f"Checksum mismatch for {dest.name}:\n"
                f"  expected {expected_md5}\n"
                f"  got      {actual}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GroupAffect-4 from Zenodo")
    parser.add_argument(
        "--record-id",
        default=DEFAULT_RECORD_ID,
        help="Zenodo record ID (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory to download files into",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Zenodo personal access token (only needed for restricted records)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip files that already exist and pass checksum (default: true)",
    )
    args = parser.parse_args()

    if args.record_id == DEFAULT_RECORD_ID:
        sys.exit(
            "No Zenodo record ID configured.\n"
            "Pass --record-id <ID> or update DEFAULT_RECORD_ID in this script."
        )

    print(f"Fetching record metadata for Zenodo record {args.record_id} …")
    record = fetch_record(args.record_id, args.token)

    files = record.get("files", [])
    if not files:
        sys.exit("No files found in this Zenodo record.")

    total_bytes = sum(f.get("size", 0) for f in files)
    print(f"Found {len(files)} file(s) — {total_bytes / 1e9:.2f} GB total\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i, file_info in enumerate(files, 1):
        key = file_info["key"]
        download_url = file_info["links"]["content"]
        checksum_field = file_info.get("checksum", "")
        expected_md5 = checksum_field.removeprefix("md5:") if checksum_field.startswith("md5:") else None

        dest = args.out_dir / key
        print(f"[{i}/{len(files)}] {key}")

        if args.skip_existing and dest.exists():
            if expected_md5 and md5(dest) == expected_md5:
                print("  already downloaded, skipping.")
                continue
            print("  exists but checksum differs, re-downloading.")

        download_file(download_url, dest, expected_md5, args.token)
        print(f"  saved to {dest}")

    print(f"\nAll files downloaded to {args.out_dir}")


if __name__ == "__main__":
    main()
