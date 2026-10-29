"""EGS cost rows for every cost year (``build_cost_data.egs_costs_for_year``).

``repo_data/costs/egs_costs.csv`` only tabulates 2030, 2040 and 2050, but a cost
table is built for every planning horizon. A year without EGS rows breaks the EGS
learning ratio in ``add_extra_components``.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_cost_data import egs_costs_for_year

pytestmark = pytest.mark.fast

EGS_CSV = Path(__file__).resolve().parents[2] / "repo_data" / "costs" / "egs_costs.csv"


@pytest.fixture
def egs_costs():
    return pd.DataFrame(
        [
            ("EGS", "capex_per_kw", 6000.0, 2030),
            ("EGS", "capex_per_kw", 5000.0, 2040),
            ("EGS", "capex_per_kw", 4000.0, 2050),
            ("EGS", "lifetime", 25.0, 2030),
            ("EGS", "lifetime", 25.0, 2040),
            ("EGS", "lifetime", 25.0, 2050),
        ],
        columns=["pypsa-name", "parameter", "value", "investment_horizon"],
    )


def _capex(df):
    return df.set_index("parameter").at["capex_per_kw", "value"]


@pytest.mark.parametrize(
    ("year", "capex"),
    [(2030, 6000.0), (2033, 5700.0), (2040, 5000.0), (2045, 4500.0), (2025, 6000.0), (2055, 4000.0)],
)
def test_capex_interpolates_between_horizons_and_clamps_outside(egs_costs, year, capex):
    assert _capex(egs_costs_for_year(egs_costs, year)) == pytest.approx(capex)


def test_output_has_the_cost_table_columns(egs_costs):
    out = egs_costs_for_year(egs_costs, 2033)
    assert list(out.columns) == ["pypsa-name", "parameter", "value"]
    assert len(out) == 2


@pytest.mark.parametrize("year", range(2024, 2051))
def test_repo_table_covers_every_cost_year(year):
    out = egs_costs_for_year(pd.read_csv(EGS_CSV), year)
    assert _capex(out) > 0
    assert out.parameter.is_unique
