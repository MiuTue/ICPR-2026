"""
Download and extract the ICPR 2026 LPR dataset.

Usage:
    uv run python download_data.py

Downloads:
    - Train set  : Google Drive ID 1yXF7Tke1JFyZNZxyL9OTEmmco_HsvrDW
    - Test set   : Dropbox shared link (public)
"""

import os
import sys
import zipfile
import urllib.request

# ── URLS ────────────────────────────────────────────────────────────────────

TRAIN_ZIP_URL = "https://drive.google.com/uc?export=download&id=1yXF7Tke1JFyZNZxyL9OTEmmco_HsvrDW"
TRAIN_ZIP_OUT = "wYe7pBJ7-train.zip"

# Dropbox direct-download link — replace dl=0 with dl=1
TEST_ZIP_URL = (
    "https://www.dropbox.com/scl/fi/os0napgfjn1ihsmwlm0fm/Pa7a3Hin-test-public.zip"
    "?rlkey=otvmdibmxuc267ljj4zwwzl4f&e=1&st=j9ryzo22&dl=1"
)
TEST_ZIP_OUT = "test-public.zip"

DATA_DIR = "data"


def _is_gdown_available():
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        return False


def _install_gdown():
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
    print("   (gdown installed)")


def download_gdrive(url: str, output: str) -> bool:
    """Download a Google Drive file using gdown (handles confirm-page)."""
    if not _is_gdown_available():
        print("   gdown not found — installing...")
        _install_gdown()

    import gdown
    print(f"   Downloading Google Drive file → {output}")
    gdown.download(url, output, quiet=False)
    return os.path.exists(output)


def download_url(url: str, output: str) -> bool:
    """Download a generic URL (Dropbox, etc.) using urllib + wget fallback."""
    print(f"   Downloading → {output}")

    # Try urllib first (works for most public URLs)
    try:
        urllib.request.urlretrieve(url, output)
        return os.path.exists(output)
    except Exception:
        pass

    # Fallback to wget (better for Dropbox which returns HTML without UA)
    import subprocess

    class WgetFailed(Exception):
        pass

    try:
        result = subprocess.run(
            [
                "wget",
                "-q",
                "--show-progress",
                "--user-agent",
                (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "-O", output,
                url,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return os.path.exists(output)
    except subprocess.CalledProcessError as e:
        print(f"   wget failed: {e.stderr}")
        raise WgetFailed(e)


def extract(zip_path: str, into: str = ".") -> None:
    """Extract a zip file safely."""
    print(f"   Extracting → {zip_path}")
    os.makedirs(into, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(into)
    print(f"   Done.")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Train set ────────────────────────────────────────────────────────────
    if not os.path.exists(TRAIN_ZIP_OUT) and not _data_exists("train"):
        print(f"\n[1/2] Downloading TRAIN set...")
        ok = download_gdrive(TRAIN_ZIP_URL, TRAIN_ZIP_OUT)
        if not ok:
            print("❌ Train download failed.")
            sys.exit(1)
    else:
        print(f"\n[1/2] Train zip already exists ({TRAIN_ZIP_OUT}) — skipping download.")

    if not _data_exists("train"):
        extract(TRAIN_ZIP_OUT, DATA_DIR)
        # Clean up zip
        os.remove(TRAIN_ZIP_OUT)
    else:
        print(f"   Train data already extracted in '{DATA_DIR}' — skipping.")

    # ── Test set ─────────────────────────────────────────────────────────────
    if not os.path.exists(TEST_ZIP_OUT) and not _data_exists("test"):
        print(f"\n[2/2] Downloading TEST set...")
        ok = download_url(TEST_ZIP_URL, TEST_ZIP_OUT)
        if not ok:
            print("❌ Test download failed.")
            sys.exit(1)
    else:
        print(f"\n[2/2] Test zip already exists ({TEST_ZIP_OUT}) — skipping download.")

    if not _data_exists("test"):
        extract(TEST_ZIP_OUT, DATA_DIR)
        os.remove(TEST_ZIP_OUT)
    else:
        print(f"   Test data already extracted in '{DATA_DIR}' — skipping.")

    print("\n✅ All data ready.")
    _report_data()


def _data_exists(subdir: str) -> bool:
    """Check if data/subdir contains actual track folders."""
    path = os.path.join(DATA_DIR, subdir)
    if not os.path.isdir(path):
        return False
    entries = os.listdir(path)
    # Heuristic: at least one track_ folder
    return any(e.startswith("track_") for e in entries)


def _report_data():
    for subdir in ("train", "test"):
        path = os.path.join(DATA_DIR, subdir)
        if os.path.isdir(path):
            tracks = [e for e in os.listdir(path) if e.startswith("track_")]
            print(f"   {subdir:5s}: {len(tracks)} track folders found")
        else:
            print(f"   {subdir:5s}: not found")


if __name__ == "__main__":
    main()
