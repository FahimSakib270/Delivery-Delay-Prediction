"""Download the public Olist CSVs without credentials or raw-data git commits."""
from __future__ import annotations
import argparse
from pathlib import Path
import tempfile
import urllib.request
import zipfile
from .config import load_config, resolve
from .data_prep import TABLE_FILES

URL = "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    destination = resolve(cfg, "paths", "raw_dir") / "olist"
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.glob("*.csv")):
        raise SystemExit("CSV files already exist. Keep the current dataset, or move it aside before downloading a new version.")
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "olist.zip"
        try:
            urllib.request.urlretrieve(URL, archive)
            with zipfile.ZipFile(archive) as z:
                expected = set(TABLE_FILES.values())
                members = {Path(name).name: name for name in z.namelist() if Path(name).name in expected}
                required = {TABLE_FILES[key] for key in ["orders", "items", "products", "customers", "sellers"]}
                if not required <= set(members):
                    raise ValueError("download does not contain the required Olist CSVs")
                for filename, member in members.items():
                    (destination / filename).write_bytes(z.read(member))
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise SystemExit(f"Download failed: {exc}. See README for manual download instructions.") from exc
    print(f"Downloaded {len(members)} tables to {destination}. Run python -m src.prepare to record checksums.")


if __name__ == "__main__":
    main()
