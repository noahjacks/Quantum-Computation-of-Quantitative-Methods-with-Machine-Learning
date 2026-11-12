"""
download.py  --  FETCH THE FREE PUBLIC DATA
===========================================

Everything this project uses is free and needs no login:

1. Open Source Bond Asset Pricing (OSBAP) "Stage 1" daily panel
   https://openbondassetpricing.com/data/
   One row per US corporate bond per trading day, built from FINRA TRACE
   (the official record of every US corporate-bond trade).  1.8 GB.
   Please cite Dickerson, Robotti & Rossetti if you use it.

2. FRED (Federal Reserve Bank of St. Louis): interest rates, credit spreads,
   VIX, inflation, unemployment ...  https://fred.stlouisfed.org

3. Kenneth French's data library: daily stock-market factor returns.

4. Federal Reserve Board: the Gilchrist-Zakrajsek "excess bond premium".

Files are downloaded once and cached under data/raw/.  Re-running is a no-op.

NOTE ON REPRODUCIBILITY: FRED only serves the last 3 years of the ICE BofA
credit-spread indices.  Our window (July-Oct 2024) is inside that range today
but will eventually fall out of it, so keep the cached copies in data/raw/fred.
"""
from __future__ import annotations

import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from . import config as C

USER_AGENT = "Mozilla/5.0 (research-data-download)"

# FRED series id -> plain-English meaning
FRED_SERIES = {
    "DGS2": "2-year US Treasury yield (%)",
    "DGS5": "5-year US Treasury yield (%)",
    "DGS10": "10-year US Treasury yield (%)",
    "DGS20": "20-year US Treasury yield (%)",
    "DGS30": "30-year US Treasury yield (%)",
    "T10Y2Y": "10-year minus 2-year Treasury yield (%), the 'curve slope'",
    "DTB3": "3-month Treasury bill rate (%)",
    "BAMLC0A0CM": "ICE BofA US investment-grade corporate spread (OAS, %)",
    "BAMLH0A0HYM2": "ICE BofA US high-yield corporate spread (OAS, %)",
    "VIXCLS": "VIX: expected stock-market volatility",
    "INDPRO": "Industrial production index (monthly)",
    "CPIAUCSL": "Consumer price index (monthly)",
    "UNRATE": "Unemployment rate (monthly, %)",
    "USEPUINDXD": "Economic policy uncertainty index (daily)",
    "DRTSCILM": "Senior loan officer survey: net % of banks tightening (quarterly)",
    "DTWEXBGS": "Trade-weighted US dollar index",
    "DCOILWTICO": "WTI crude oil price ($/barrel)",
}

FRENCH_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
              "F-F_Research_Data_Factors_daily_CSV.zip")
EBP_URL = "https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv"


def _fetch(url: str, timeout: int = 45, retries: int = 4,
           user_agent: str | None = USER_AGENT) -> bytes:
    """Download a URL and return the raw bytes (with a few polite retries).

    ``user_agent=None`` sends Python's default client name.  FRED silently
    stalls on browser-like client names, so FRED downloads use the default.
    """
    last_error: Exception | None = None
    headers = {"User-Agent": user_agent} if user_agent else {}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # network hiccup: wait and retry
            last_error = exc
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def download_fred(force: bool = False) -> None:
    out_dir = C.DATA_RAW / "fred"
    out_dir.mkdir(parents=True, exist_ok=True)
    for series in FRED_SERIES:
        target = out_dir / f"{series}.csv"
        if target.exists() and not force:
            continue
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=2019-01-01"
        data = _fetch(url, user_agent=None)
        if not data.startswith(b"observation_date"):
            raise RuntimeError(f"Unexpected FRED response for {series}")
        target.write_bytes(data)
        print(f"  FRED {series:<13} -> {target.name}  ({len(data):,} bytes)", flush=True)
        time.sleep(0.3)  # be polite to the server


def download_french(force: bool = False) -> None:
    out_dir = C.DATA_RAW / "french"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "F-F_Research_Data_Factors_daily.csv"
    if target.exists() and not force:
        return
    raw = _fetch(FRENCH_URL)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        target.write_bytes(zf.read(name))
    print(f"  French daily factors -> {target.name}")


def download_ebp(force: bool = False) -> None:
    out_dir = C.DATA_RAW / "ebp"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "ebp_csv.csv"
    if target.exists() and not force:
        return
    data = _fetch(EBP_URL)
    if not data.startswith(b"date"):
        raise RuntimeError("Unexpected response for the excess bond premium file")
    target.write_bytes(data)
    print(f"  Excess bond premium -> {target.name}")


def download_osbap(force: bool = False) -> None:
    """The big one (1.8 GB).  Streams to disk, then unzips."""
    target = C.OSBAP_PARQUET
    if target.exists() and not force:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    zip_path = target.with_suffix(".zip")
    print("  Downloading the OSBAP bond panel (1.8 GB) - this takes a few minutes ...")
    req = urllib.request.Request(C.OSBAP_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, open(zip_path, "wb") as fh:
        while True:
            chunk = resp.read(1 << 22)
            if not chunk:
                break
            fh.write(chunk)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target.parent)
    print(f"  OSBAP panel -> {target}")


def download_all(force: bool = False) -> None:
    C.ensure_dirs()
    print("Downloading free public data (cached files are skipped):")
    download_osbap(force)
    download_fred(force)
    download_french(force)
    download_ebp(force)
    print("Done.")


if __name__ == "__main__":
    download_all(force="--force" in sys.argv)
