"""Shared set-up for the tests: load the real bond panel once per test run."""
import pytest

from qon import config as C


@pytest.fixture(scope="session")
def panel():
    if not C.OSBAP_PARQUET.exists():
        pytest.skip("real data not downloaded (run scripts/01_download_data.py)")
    from qon.panel import load_panel
    return load_panel(verbose=False)


@pytest.fixture(scope="session")
def market(panel):
    from qon.market import build_market
    return build_market(panel)


@pytest.fixture(scope="session")
def events():
    if not (C.DATA_PROCESSED / "events_main.parquet").exists():
        pytest.skip("dataset not built (run scripts/02_build_dataset.py)")
    from qon.dataset import load_events
    return load_events()
