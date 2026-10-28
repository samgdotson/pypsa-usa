"""Tests for EER demand profile loading."""

import numpy as np
import pandas as pd
import pytest
import tables
from build_demand import ReadEer


def _write_eer_fixture(path, model_years=(2030,), scale=None, reverse=()):
    """Write an EER-like h5 file; ``scale`` multiplies a year's profiles and ``reverse`` flips them in time."""
    hours = ReadEer.HOURS_PER_YEAR
    total_hours = hours * len(ReadEer.WEATHER_YEARS)
    scale = scale or {}

    with tables.open_file(path, "w") as h5:
        for model_year in model_years:
            group = h5.create_group("/", str(model_year))
            h5.create_array(
                group,
                "columns",
                np.array([b"datetime", b"AL", b"CA"]),
            )
            h5.create_carray(
                group,
                "datetime",
                tables.StringAtom(itemsize=30),
                shape=(total_hours,),
            )
            for state, offset in {"AL": 0, "CA": 1000000}.items():
                values = np.arange(offset, offset + total_hours, dtype=np.int32) * scale.get(model_year, 1)
                if model_year in reverse:
                    values = values[::-1]
                node = h5.create_carray(
                    group,
                    state,
                    tables.Int32Atom(),
                    shape=(total_hours,),
                )
                node[:] = values


def _read(path, horizons, weather_year=2010):
    demand = ReadEer(str(path), planning_horizons=horizons, renewable_weather_years=[weather_year]).read_demand()
    return demand.droplevel(["sector", "subsector", "fuel"])


def test_read_eer_selects_model_and_weather_year_with_cst_shift(tmp_path):
    path = tmp_path / "eer.h5"
    _write_eer_fixture(path, model_years=(2030,))

    reader = ReadEer(
        str(path),
        planning_horizons=[2030],
        renewable_weather_years=[2010],
    )
    demand = reader.read_demand()

    weather_start = ReadEer.WEATHER_YEARS.index(2010) * ReadEer.HOURS_PER_YEAR
    first_snapshot = pd.Timestamp("2030-01-01 00:00:00")
    sixth_snapshot = pd.Timestamp("2030-01-01 06:00:00")

    assert len(demand) == ReadEer.HOURS_PER_YEAR
    assert demand.index.names == ["snapshot", "sector", "subsector", "fuel"]
    assert "Alabama" in demand.columns
    assert "California" in demand.columns
    assert demand.loc[(first_snapshot, "all", "all", "electricity"), "Alabama"] == (
        weather_start + ReadEer.HOURS_PER_YEAR - ReadEer.CST_TO_UTC_SHIFT
    )
    assert demand.loc[(sixth_snapshot, "all", "all", "electricity"), "Alabama"] == weather_start


def test_read_eer_rejects_invalid_weather_year(tmp_path):
    with pytest.raises(ValueError, match="supports weather years"):
        ReadEer(str(tmp_path / "eer.h5"), planning_horizons=[2030], renewable_weather_years=[2014])


def test_read_eer_rejects_multiple_weather_years(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        ReadEer(str(tmp_path / "eer.h5"), planning_horizons=[2030], renewable_weather_years=[2010, 2011])


@pytest.mark.parametrize("year", [2019, 2020, 2051])
def test_read_eer_rejects_horizon_outside_model_years(tmp_path, year):
    with pytest.raises(ValueError, match="unsupported year"):
        ReadEer(str(tmp_path / "eer.h5"), planning_horizons=[year], renewable_weather_years=[2010])


@pytest.mark.parametrize(
    ("year", "shape_year"),
    [(2021, 2021), (2022, 2021), (2023, 2021), (2024, 2025), (2027, 2025), (2028, 2030), (2049, 2050)],
)
def test_shape_year_is_nearest_model_year_earlier_on_tie(year, shape_year):
    assert ReadEer._shape_year(year) == shape_year


def test_unmodeled_year_interpolates_annual_totals_per_state(tmp_path):
    path = tmp_path / "eer.h5"
    _write_eer_fixture(path, model_years=(2025, 2030), scale={2030: 2})

    modeled = _read(path, [2025, 2030])
    totals = modeled.groupby(modeled.index.year).sum()
    demand = _read(path, [2027])

    expected = totals.loc[2025] + (totals.loc[2030] - totals.loc[2025]) * 2 / 5
    pd.testing.assert_series_equal(demand.sum(), expected, check_names=False)


@pytest.mark.parametrize(("year", "shape_year"), [(2027, 2025), (2028, 2030)])
def test_unmodeled_year_reuses_nearest_model_year_shape(tmp_path, year, shape_year):
    path = tmp_path / "eer.h5"
    # 2030 runs backwards in time, so the two candidate shapes differ
    _write_eer_fixture(path, model_years=(2025, 2030), scale={2030: 2}, reverse=(2030,))

    shape = _read(path, [shape_year])
    demand = _read(path, [year])

    assert len(demand) == ReadEer.HOURS_PER_YEAR
    assert (demand.index.year == year).all()
    np.testing.assert_allclose(
        (demand / demand.sum()).to_numpy(),
        (shape / shape.sum()).to_numpy(),
    )


def test_mixed_horizons_keep_model_years_unchanged(tmp_path):
    path = tmp_path / "eer.h5"
    _write_eer_fixture(path, model_years=(2025, 2030), scale={2030: 2})

    alone = _read(path, [2025])
    mixed = _read(path, [2025, 2026, 2030])

    assert sorted(mixed.index.year.unique()) == [2025, 2026, 2030]
    pd.testing.assert_frame_equal(mixed[mixed.index.year == 2025], alone, check_dtype=False, check_freq=False)
