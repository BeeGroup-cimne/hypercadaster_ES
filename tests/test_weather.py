"""Climate statistics and EPW export from a synthetic ERA5-Land year."""
import hypercadaster_ES as hc
from hypercadaster_ES import weather


def test_weather_statistics_degree_days(weather_year_df):
    stats = hc.weather.weather_statistics(weather_year_df, hdd_base=18.0, cdd_base=21.0)
    # constant 10 C -> HDD = (18-10) * 366 days, CDD = 0
    assert stats["CDD"] == 0.0
    assert stats["HDD"] == round(8.0 * stats["n_days"], 1)
    assert stats["t_mean"] == 10.0
    assert stats["n_days"] == 366  # 2020 is a leap year


def test_weather_statistics_monthly(weather_year_df):
    stats = hc.weather.weather_statistics(weather_year_df, by_month=True)
    monthly = stats["monthly"]
    assert monthly.height == 12
    assert {"month", "HDD", "CDD"}.issubset(set(monthly.columns))


def test_write_epw_is_8760(weather_year_df, tmp_path):
    path = tmp_path / "test.epw"
    weather.write_epw(
        weather_year_df, str(path),
        year=2020, latitude=41.4, longitude=2.2, elevation=12.0,
    )
    assert path.exists()
    lines = path.read_text().splitlines()
    # 8 header rows + 8760 hourly data rows (Feb 29 dropped)
    assert len(lines) == 8 + 8760
    assert lines[0].startswith("LOCATION,")
    # data rows begin with the year
    assert lines[8].split(",")[0] == "2020"
