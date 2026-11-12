"""
STEP 1 - Download the free public data (about 1.8 GB, done once).

    .venv/bin/python scripts/01_download_data.py

See src/qon/download.py for where each file comes from.
"""
import _path  # noqa: F401  (makes `import qon` work when run as a script)
from qon.download import download_all

if __name__ == "__main__":
    download_all()
