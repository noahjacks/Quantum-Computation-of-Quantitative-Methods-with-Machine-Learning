"""
STEP 2 - Turn the raw bond data into the table of replayed quote requests.

    .venv/bin/python scripts/02_build_dataset.py      (about 1 minute)

Writes data/processed/events_main.parquet (16,000 requests in HSBC's window),
events_calibration.parquet (8,000 requests from the 40 days before) and
docs/FEATURE_COVERAGE.md (what happened to each of the 216 features).
"""
import _path  # noqa: F401
from qon.dataset import build_events
from qon.feature_catalog import write_coverage_report

if __name__ == "__main__":
    build_events()
    table = write_coverage_report()
    print(f"{(table['status'] == 'BUILT').sum()} of {len(table)} features built "
          "-> docs/FEATURE_COVERAGE.md")
