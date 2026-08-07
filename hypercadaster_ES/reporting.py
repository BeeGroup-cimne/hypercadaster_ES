"""Interactive HTML reports for the occupant-behaviour simulations.

:func:`scenario_report` turns what ``energy.run_behavior_scenarios`` produces —
the per-use-type KPI frame, and the hourly EnergyPlus CSVs left behind in its
output directory — into one self-contained page: the summarised comparisons that
answer "which occupant costs what", and the raw hourly series behind them for
when a summary looks wrong and you need to see the year.

The page carries no external assets (Plotly is inlined), so it opens from disk
and survives being emailed.
"""
from __future__ import annotations

import glob
import html
import os
import re

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
# Categorical hues assigned in FIXED order to the scenarios, so a scenario keeps
# its colour across every chart on the page and between runs — colour follows the
# entity, never its rank or its position in a filtered list. Both columns are
# selected for their own surface; the dark one is the same hues re-stepped, not an
# automatic inversion. Validated (lightness band, chroma floor, CVD separation,
# normal-vision floor) on the adjacent pairlist these bar/line forms use.
_SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                 "#4a3aa7", "#e34948")
_SERIES_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300",
                "#9085e9", "#e66767")

# Season order + the ink used for the recessive furniture.
_SEASON_ORDER = ("heating", "in_between", "cooling")
_INK = {"light": {"surface": "#fcfcfb", "primary": "#0b0b0b",
                  "secondary": "#52514e", "muted": "#6f6e6a", "grid": "#e4e3df"},
        "dark": {"surface": "#1a1a19", "primary": "#ffffff",
                 "secondary": "#c3c2b7", "muted": "#93928a", "grid": "#333230"}}

# Scenario display order: the normative case last, so the occupant behaviours read
# against each other first.
_SCENARIO_ORDER = ("unaware", "unaware_ff", "random_mix", "aware", "aware_ff",
                   "normative", "cte")

def _nonresidential_run():
    """KPI scenario label → the run it was summarised from, where the two differ.

    The non-residential zones run the SAME normative user in every scenario, so the
    KPI builder reports them once, under ``"normative"``, out of one run; the raw
    hourly section has to reach for that same run rather than for a directory of
    that name, which does not exist.
    """
    from hypercadaster_ES.energy import (NONRESIDENTIAL_SCENARIO_LABEL,
                                         NONRESIDENTIAL_SOURCE_SCENARIO)
    return {NONRESIDENTIAL_SCENARIO_LABEL: NONRESIDENTIAL_SOURCE_SCENARIO}


def _ordered(values, order):
    """``values`` sorted by ``order``, with anything unknown appended in place."""
    known = [v for v in order if v in set(values)]
    return known + [v for v in dict.fromkeys(values) if v not in set(order)]


def _scenario_colors(scenarios, mode="light"):
    ramp = _SERIES_LIGHT if mode == "light" else _SERIES_DARK
    return {s: ramp[i % len(ramp)] for i, s in enumerate(scenarios)}


def _scenario_dash(scenario):
    """Line style for a scenario — dotted for the free-floating twins.

    A second encoding beside the hue, and not only for colour vision: a twin
    often traces its conditioned partner exactly (identical shading, identical
    gains), and one line then hides the other completely. Dotted, both are
    visible where they coincide.
    """
    return "dot" if str(scenario).endswith("_ff") else "solid"


# --------------------------------------------------------------------------- #
# Raw hourly outputs
# --------------------------------------------------------------------------- #
def _read_scenario_hourly(output_dir, building_reference=None):
    """``{scenario: DataFrame}`` of the hourly EnergyPlus output of each run.

    ``run_behavior_scenarios`` leaves one directory per scenario next to its
    results (``{building}_{scenario}/eplusout.csv``); this picks them up so the
    report can show what the KPI rows were summarised from.
    """
    from hypercadaster_ES.energy import _read_eplus_csv
    out = {}
    if not output_dir or not os.path.isdir(output_dir):
        return out
    pattern = f"{building_reference}_*" if building_reference else "*"
    for d in sorted(glob.glob(os.path.join(output_dir, pattern))):
        if not os.path.isdir(d):
            continue
        csv = os.path.join(d, "eplusout.csv")
        if not os.path.exists(csv):
            continue
        name = os.path.basename(d)
        if building_reference and name.startswith(f"{building_reference}_"):
            name = name[len(building_reference) + 1:]
        df = _read_eplus_csv(csv)
        if df is not None and len(df):
            out[name] = df
    return out


def _co2_outdoor_ppm():
    """The outdoor CO2 background the zones decay towards (config ``co2``)."""
    from hypercadaster_ES.energy import _CO2
    return float((_CO2 or {}).get("outdoor_ppm", 420))


def _hourly_building_series(df, areas=None, zones=None):
    """Collapse one run's per-zone hourly output to BUILDING series.

    Returns a frame with the timestamp parts plus ``t_indoor`` (floor-area
    weighted mean operative temperature — the building's temperature, not an
    average of unequal rooms), ``t_outdoor``, ``hvac_kw`` and ``solar_kw``.

    ``zones`` restricts the collapse to one set of zones — the report uses it to
    keep each USE apart. Mixing them makes the building series answer nobody's
    question: a 144 m² parking box at 1 ren/h and no occupants pulled the air-change
    curve of a dwelling up by a tenth and its CO2 down towards the outdoor
    background, so neither number described the parking or the flats.
    """
    # "ENVIRONMENT:" prefixes the site weather, which belongs to every use.
    #
    # The output KEY is usually the zone name, but NOT for the ideal-loads
    # variables: ExpandObjects reports those under the system it built, "H_F1 IDEAL
    # LOADS AIR SYSTEM:Zone Ideal Loads Supply Air Total Heating Energy". Matching
    # the key exactly therefore dropped every HVAC column the moment ``zones`` was
    # given — which is always, since the report keeps each use apart — and the
    # daily HVAC series was a flat zero for every scenario. A key that STARTS WITH
    # a kept zone name at a space boundary belongs to that zone; this is the same
    # rule ``energy._zone_column`` uses to find the columns for the KPIs, which is
    # why the KPI table showed a demand the plot did not.
    keep = ({str(z).upper() for z in zones} | {"ENVIRONMENT"}) if zones else None

    def _kept(column):
        if keep is None or ":" not in column:
            return True
        key = column.split(":")[0].strip().upper()
        return key in keep or any(key.startswith(z + " ") for z in keep)

    cols = [c for c in df.columns if _kept(c)]

    def _match(suffix, exclude=None):
        hits = [c for c in cols if suffix.lower() in c.lower()]
        if exclude:
            hits = [c for c in hits if exclude.lower() not in c.lower()]
        return hits

    temp_cols = _match("Zone Operative Temperature")
    inf_cols = _match("Infiltration Standard Density Air Change Rate")
    ven_cols = _match("Ventilation Standard Density Air Change Rate")
    heat_cols = _match("Ideal Loads Supply Air Total Heating Energy")
    cool_cols = _match("Ideal Loads Supply Air Total Cooling Energy")
    sol_cols = _match("Transmitted Solar Radiation Energy")
    co2_cols = _match("Zone Air CO2 Concentration")
    hum_cols = _match("Zone Mean Air Humidity Ratio")
    occ_cols = _match("Zone People Occupant Count")
    out_col = next((c for c in cols if "Site Outdoor Air Drybulb" in c), None)

    n = len(df)
    zone_area = areas or {}

    def _weighted_mean(columns):
        if not columns:
            return np.full(n, np.nan)
        weights, series = [], []
        for c in columns:
            key = c.split(":")[0].strip().upper()
            weights.append(float(zone_area.get(key, 1.0)))
            series.append(df[c].to_numpy(dtype=float))
        w = np.asarray(weights, dtype=float)
        if w.sum() <= 0:
            w = np.ones_like(w)
        return (np.vstack(series) * w[:, None]).sum(axis=0) / w.sum()

    def _sum_kwh(columns):
        if not columns:
            return np.zeros(n)
        return np.vstack([df[c].to_numpy(dtype=float) for c in columns]).sum(axis=0) / 3.6e6

    return pd.DataFrame({
        "month": df["month"].to_numpy(),
        "day": df["day"].to_numpy(),
        "hour": df["hour"].to_numpy(),
        "t_indoor": _weighted_mean(temp_cols),
        "t_outdoor": df[out_col].to_numpy(dtype=float) if out_col else np.full(n, np.nan),
        "hvac_kwh": _sum_kwh(heat_cols) + _sum_kwh(cool_cols),
        "solar_kwh": _sum_kwh(sol_cols),
        "co2_ppm": _weighted_mean(co2_cols),
        # Humidity ratio (kg water / kg dry air) — the psychrometric ordinate the
        # comfort chart plots against the operative temperature. NaN on a run made
        # before it was requested, which the chart detects and skips.
        "w_kgkg": _weighted_mean(hum_cols),
        # People actually in the building, summed across its zones.
        "occupants": (np.vstack([df[c].to_numpy(dtype=float) for c in occ_cols])
                      .sum(axis=0) if occ_cols else np.full(n, np.nan)),
        # What the occupants added on top of the outdoor background: the
        # generation term, which is what ventilation has to remove.
        "co2_excess_ppm": _weighted_mean(co2_cols) - _co2_outdoor_ppm(),
        # Total OUTDOOR-air change: what flushes the CO2 above.
        "ach_total": (np.nan_to_num(_weighted_mean(inf_cols), nan=0.0)
                      + np.nan_to_num(_weighted_mean(ven_cols), nan=0.0)),
    })


_WEEKDAY_ABBR = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _weekday_stamp(month, day, hour, year=None):
    """``"Thu 12-03 08h"`` — the weekday is what makes a weekly occupancy rhythm
    legible, and it is exactly what a bare date hides."""
    label = f"{int(month):02d}-{int(day):02d} {int(hour):02d}h"
    try:
        import datetime as _dt
        if year:
            return f"{_WEEKDAY_ABBR[_dt.date(int(year), int(month), int(day)).weekday()]} {label}"
    except (ValueError, TypeError):
        pass
    return label


def _daily(series):
    """Daily aggregate of an hourly building series: mean temperatures, summed
    energies, plus an ordinal day index for plotting."""
    g = series.groupby(["month", "day"], as_index=False).agg(
        t_indoor=("t_indoor", "mean"), t_outdoor=("t_outdoor", "mean"),
        hvac_kwh=("hvac_kwh", "sum"), solar_kwh=("solar_kwh", "sum"),
        co2_ppm=("co2_ppm", "mean"), ach_total=("ach_total", "mean"),
        occupants=("occupants", "mean"), co2_excess_ppm=("co2_excess_ppm", "mean"))
    g = g.sort_values(["month", "day"]).reset_index(drop=True)
    g["date"] = [f"{int(m):02d}-{int(d):02d}" for m, d in zip(g["month"], g["day"])]
    return g


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _fig_shell(fig, mode="light", height=380, ylab="", xlab="", showlegend=True):
    """Recessive furniture: transparent surface (the page supplies it), thin
    axes, muted grid, legend above the plot. Text wears ink tokens, never a
    series colour."""
    ink = _INK[mode]
    # The legend goes BELOW the plot. Above, it shares a band with Plotly's
    # floating modebar, and a legend wide enough to wrap runs straight under the
    # toolbar — the last series label becomes unreadable exactly when there are
    # enough series for the legend to matter.
    fig.update_layout(
        height=height, margin=dict(l=64, r=24, t=24, b=76 if showlegend else 44),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="ui-sans-serif, system-ui, -apple-system, Segoe UI, "
                         "Roboto, Helvetica, Arial", size=13, color=ink["secondary"]),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0,
                    font=dict(color=ink["secondary"]), bgcolor="rgba(0,0,0,0)"),
        showlegend=showlegend,
        hoverlabel=dict(font_size=12), hovermode="closest")
    fig.update_xaxes(title_text=xlab, showgrid=False, zeroline=False,
                     linecolor=ink["grid"], ticks="outside", tickcolor=ink["grid"],
                     title_font=dict(color=ink["muted"]),
                     tickfont=dict(color=ink["secondary"]))
    fig.update_yaxes(title_text=ylab, gridcolor=ink["grid"], zeroline=False,
                     linecolor="rgba(0,0,0,0)", ticks="",
                     title_font=dict(color=ink["muted"]),
                     tickfont=dict(color=ink["secondary"]))
    return fig


def _bar_by_scenario_season(kpis, value, ylab, mode="light", fmt="{:.1f}"):
    """Grouped bars: one bar per scenario within each season.

    Magnitude compared across a small set of named things → bars from a zero
    baseline, grouped by season so the seasons are the reading order and the
    scenarios are the comparison. Every bar is directly labelled, which is also
    the relief the palette's contrast WARN requires.
    """
    import plotly.graph_objects as go
    seasons = _ordered(kpis["season"].unique(), _SEASON_ORDER)
    scenarios = _ordered(kpis["scenario"].unique(), _SCENARIO_ORDER)
    colors = _scenario_colors(scenarios, mode)
    fig = go.Figure()
    for s in scenarios:
        sub = kpis[kpis["scenario"] == s].set_index("season")
        vals = [float(sub[value].get(se, np.nan)) if se in sub.index else np.nan
                for se in seasons]
        fig.add_trace(go.Bar(
            name=s, x=seasons, y=vals, marker_color=colors[s],
            marker_line_width=0,
            text=[("" if v != v else fmt.format(v)) for v in vals],
            textposition="outside", textfont=dict(size=11, color=_INK[mode]["muted"]),
            cliponaxis=False,
            hovertemplate=f"<b>{s}</b><br>%{{x}}<br>{ylab}: %{{y:.2f}}<extra></extra>"))
    # 2 px of surface between adjacent bars, per the mark spec.
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    return _fig_shell(fig, mode, ylab=ylab)


def _fig_distribution_bands(kpis, prefix="t", unit=" °C", mode="light", fmt="{:.1f}",
                            center="p50"):
    """Distribution of a per-row metric per scenario, day and night.

    Four nested spans rather than one: min–max as a hairline, p10–p90, p20–p80,
    and a centre marker. The extremes are the point of having them — a dwelling
    that touches 12 °C once is a different proposition from one whose p10 is
    12 °C, and a percentile band alone cannot tell you which you have. The same
    reading applies to the air change rate, where one number hides a dwelling
    sealed for twenty hours and flung open for one.

    ``prefix`` selects the column family (``t_…`` or ``ach_total_…``); ``center``
    the statistic the marker shows. Temperatures take the median; the OUTDOOR-AIR
    rates take the MEAN and never a median — an ACH is a flow, and the median of a
    series that is 0 for twenty hours out of twenty-one is 0, which describes a
    dwelling that never opens a window rather than the air it actually renewed.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    seasons = _ordered(kpis["season"].unique(), _SEASON_ORDER)
    scenarios = _ordered(kpis["scenario"].unique(), _SCENARIO_ORDER)
    colors = _scenario_colors(scenarios, mode)
    ink = _INK[mode]
    parts = ("day", "night")
    # widest span first so the narrower ones draw on top of it
    spans = (("min", "max", 1.5, 0.9), ("p10", "p90", 5, 0.55), ("p20", "p80", 11, 1.0))
    fig = make_subplots(rows=1, cols=len(seasons), shared_yaxes=True,
                        subplot_titles=[s.replace("_", " ") for s in seasons],
                        horizontal_spacing=0.05)
    seen = set()
    for ci, season in enumerate(seasons, start=1):
        for s in scenarios:
            row = kpis[(kpis["scenario"] == s) & (kpis["season"] == season)]
            if row.empty:
                continue
            row = row.iloc[0]
            for part in parts:
                stats = {k: row.get(f"{prefix}_{k}_{part}")
                         for k in ("min", "p10", "p20", center, "p80", "p90", "max")}
                if stats[center] is None or stats[center] != stats[center]:
                    continue
                label = f"{s} · {part}"
                faded = 1.0 if part == "day" else 0.5
                clabel = "mean" if center == "mean" else "median"
                hover = ("<b>%s</b> · %s<br>max %s<br>p90 %s · p80 %s<br>"
                         f"<b>{clabel} %s</b><br>"
                         "p20 %s · p10 %s<br>min %s"
                         "<extra></extra>") % (
                    s, part, *[("—" if stats[k] is None or stats[k] != stats[k]
                                else fmt.format(stats[k]) + unit)
                               for k in ("max", "p90", "p80", center, "p20", "p10", "min")])
                for lo_k, hi_k, width, alpha in spans:
                    lo, hi = stats[lo_k], stats[hi_k]
                    if any(v is None or v != v for v in (lo, hi)):
                        continue
                    fig.add_trace(go.Scatter(
                        x=[lo, hi], y=[label, label], mode="lines",
                        line=dict(color=colors[s], width=width),
                        opacity=alpha * faded, name=s, legendgroup=s,
                        showlegend=(s not in seen and lo_k == "p20"),
                        hovertemplate=hover), row=1, col=ci)
                    if lo_k == "p20":
                        seen.add(s)
                fig.add_trace(go.Scatter(
                    x=[stats[center]], y=[label], mode="markers",
                    marker=dict(color=ink["surface"], size=8,
                                line=dict(color=colors[s], width=2)),
                    opacity=faded, showlegend=False, hovertemplate=hover),
                    row=1, col=ci)
    # No per-axis title: faceting repeats it under every panel, where the three
    # copies collide. The unit rides on the ticks instead.
    fig = _fig_shell(fig, mode, height=max(360, 30 * len(scenarios) * 2))
    fig.update_xaxes(ticksuffix=unit)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    for a in fig.layout.annotations:
        a.font.color = ink["muted"]
        a.font.size = 12
    return fig


def _categorical_time_axis(fig, n_labels=12):
    """Treat the ``MM-DD`` / ``MM-DD HHh`` stamps as CATEGORIES, not dates.

    Plotly reads "01-02" as a date and renders the axis in YEARS (2001, 2002, …),
    which is both wrong and confusing. A category axis also handles a run period
    that is not contiguous — simulating January and July leaves a gap no date axis
    would draw honestly.
    """
    fig.update_xaxes(type="category", nticks=n_labels, tickangle=0)
    return fig


def _fig_daily_series(hourly, value, ylab, mode="light", with_outdoor=False):
    """Daily series per scenario — change over time, so lines, one per scenario."""
    import plotly.graph_objects as go
    scenarios = _ordered(list(hourly), _SCENARIO_ORDER)
    colors = _scenario_colors(scenarios, mode)
    ink = _INK[mode]
    fig = go.Figure()
    if with_outdoor and scenarios:
        d0 = _daily(hourly[scenarios[0]])
        fig.add_trace(go.Scatter(
            x=d0["date"], y=d0["t_outdoor"], mode="lines", name="outdoor",
            line=dict(color=ink["muted"], width=1.5, dash="dot"),
            hovertemplate="outdoor %{y:.1f} °C<extra></extra>"))
    for s in scenarios:
        d = _daily(hourly[s])
        fig.add_trace(go.Scatter(
            x=d["date"], y=d[value], mode="lines", name=s,
            line=dict(color=colors[s], width=2, dash=_scenario_dash(s)),
            hovertemplate=f"<b>{s}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>"))
    fig = _fig_shell(fig, mode, ylab=ylab)
    fig.update_layout(hovermode="x unified")
    return _categorical_time_axis(fig)


def _fig_weekly_profile(hourly, value, ylab, mode="light", year=None, fmt="{:.2f}"):
    """The average WEEK at hourly resolution — 168 points, Monday 00h to Sunday 23h.

    Occupancy is a weekly rhythm, not an annual one: a daily mean flattens the very
    shape that matters (who is home at 03:00, who leaves at nine, how Saturday
    differs from Tuesday) into one number a day, and the year-long x-axis then
    hides what is left. Averaging each hour-of-week across the whole run keeps the
    rhythm and still shows the level.
    """
    import plotly.graph_objects as go
    import datetime as _dt
    scenarios = _ordered(list(hourly), _SCENARIO_ORDER)
    colors = _scenario_colors(scenarios, mode)
    fig = go.Figure()
    labels = [f"{_WEEKDAY_ABBR[d]} {h:02d}h" for d in range(7) for h in range(24)]
    for s in scenarios:
        d = hourly[s]
        if value not in d:
            continue
        idx = []
        for m, dd, hh in zip(d["month"], d["day"], d["hour"]):
            try:
                wd = _dt.date(int(year or 2021), int(m), int(dd)).weekday()
            except (ValueError, TypeError):
                wd = 0
            idx.append(wd * 24 + min(23, max(0, int(hh) - 1)))
        prof = (pd.DataFrame({"i": idx, "v": d[value].to_numpy(dtype=float)})
                .groupby("i")["v"].mean().reindex(range(168)))
        fig.add_trace(go.Scatter(
            x=labels, y=prof.to_numpy(), mode="lines", name=s,
            line=dict(color=colors[s], width=2, dash=_scenario_dash(s)),
            hovertemplate=f"<b>{s}</b><br>%{{x}}<br>%{{y:{fmt[2:-1]}}}<extra></extra>"))
    fig = _fig_shell(fig, mode, ylab=ylab)
    fig.update_layout(hovermode="x unified")
    # Day boundaries, so the week reads as seven days rather than 168 ticks.
    ink = _INK[mode]
    for d in range(1, 7):
        fig.add_vline(x=d * 24 - 0.5, line=dict(color=ink["muted"], width=1,
                                                dash="dot"), opacity=0.5)
    fig.update_xaxes(type="category", tickmode="array",
                     tickvals=[d * 24 + 12 for d in range(7)],
                     ticktext=list(_WEEKDAY_ABBR), tickangle=0)
    return fig


def _worst_week(daily, kind="cooling", days=7):
    """The ``days``-long window with the most extreme outdoor temperature — the
    hottest stretch for ``cooling``, the coldest for ``heating``. Returns the set
    of ``(month, day)`` pairs, or ``None`` when the run is shorter than a window."""
    if daily is None or daily.empty:
        return None
    roll = daily["t_outdoor"].rolling(days, min_periods=max(1, days // 2)).mean()
    if roll.isna().all():
        return None
    end = int(roll.idxmax() if kind == "cooling" else roll.idxmin())
    lo = max(0, end - days + 1)
    return set(zip(daily["month"][lo:end + 1], daily["day"][lo:end + 1]))


def _worst_co2_week(daily, days=7):
    """The ``days``-long window with the highest mean indoor CO2 — where the air
    renewal is least able to keep up with the occupants."""
    if daily is None or daily.empty or "co2_ppm" not in daily:
        return None
    roll = daily["co2_ppm"].rolling(days, min_periods=max(1, days // 2)).mean()
    if roll.isna().all():
        return None
    end = int(roll.idxmax())
    lo = max(0, end - days + 1)
    return set(zip(daily["month"][lo:end + 1], daily["day"][lo:end + 1]))


def _fig_co2_week(hourly, mode="light", days=7, year=None, window=None):
    """A week of HOURLY indoor CO2 with the air change that produced it.

    Daily means cannot answer the question CO2 raises — whether the air recovers
    between occupied periods — because the overnight build-up and the morning
    flush average to the same number as a flat mediocre day. The panels share a
    time axis so a spike can be read straight down onto the ventilation (or the
    absence of it) at that hour, and the weekday ticks make the weekly rhythm of
    the occupancy agenda legible.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    scenarios = _ordered(list(hourly), _SCENARIO_ORDER)
    if not scenarios:
        return None
    colors = _scenario_colors(scenarios, mode)
    ink = _INK[mode]
    if window is None:
        window = _worst_co2_week(_daily(hourly[scenarios[0]]), days)
    if not window:
        return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.58, 0.42])
    drew = False
    for s in scenarios:
        h = hourly[s]
        keep = pd.Series([(m, d) in window for m, d in zip(h["month"], h["day"])],
                         index=h.index)
        sub = h[keep]
        if sub.empty or "co2_ppm" not in sub:
            continue
        x = [_weekday_stamp(m, d, hr, year)
             for m, d, hr in zip(sub["month"], sub["day"], sub["hour"])]
        fig.add_trace(go.Scatter(
            x=x, y=sub["co2_ppm"], mode="lines", name=s, legendgroup=s,
            line=dict(color=colors[s], width=2, dash=_scenario_dash(s)),
            hovertemplate=f"<b>{s}</b><br>%{{x}}<br>%{{y:.0f}} ppm<extra></extra>"),
            row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=sub["ach_ppm"] if "ach_ppm" in sub else sub["ach_total"],
            mode="lines", name=s, legendgroup=s, showlegend=False,
            line=dict(color=colors[s], width=2, dash=_scenario_dash(s)),
            hovertemplate=f"<b>{s}</b><br>%{{x}}<br>%{{y:.2f}} ACH<extra></extra>"),
            row=2, col=1)
        drew = True
    if not drew:
        return None
    fig = _fig_shell(fig, mode, height=560)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(type="category", nticks=8, tickangle=0)
    fig.update_yaxes(title_text="CO₂ ppm", row=1, col=1)
    fig.update_yaxes(title_text="air change ACH", row=2, col=1)
    return fig


# --------------------------------------------------------------------------- #
# Psychrometry — the indoor state as a point in (temperature, humidity)
# --------------------------------------------------------------------------- #
# The psychrometry itself lives in ``energy`` with the rest of the physics — the
# report draws the model, it does not define it, and the chart's comfort field has
# to be the SAME object the KPI columns are computed from or the page contradicts
# its own table.
def _psy():
    from hypercadaster_ES import energy
    return energy


def _p_sat(t_c):
    return _psy()._p_sat(t_c)


def _humidity_ratio(t_c, rh):
    return _psy()._humidity_ratio(t_c, rh)


def _relative_humidity(t_c, w):
    return _psy()._relative_humidity(t_c, w)


def _rgba(hex_color, alpha):
    h = str(hex_color).lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _bin_psy_points(t, w, t_step=0.25, w_step=0.25):
    """Collapse hourly ``(temperature, humidity ratio g/kg)`` pairs onto a grid,
    returning ``(t, w, hours)`` per occupied cell.

    A season is ~2 000 occupied hours per scenario, and six clouds of that at full
    resolution is both unreadable and a several-MB page. Binning keeps the SHAPE of
    the distribution — which is the whole point of a psychrometric plot — while
    making the density explicit (hours per cell) instead of leaving it to however
    much ink happens to pile up.
    """
    if not len(t):
        return np.zeros(0), np.zeros(0), np.zeros(0)
    ti = np.round(np.asarray(t, dtype=float) / t_step).astype(np.int64)
    wi = np.round(np.asarray(w, dtype=float) / w_step).astype(np.int64)
    keys, counts = np.unique(np.stack([ti, wi], axis=1), axis=0, return_counts=True)
    return keys[:, 0] * t_step, keys[:, 1] * w_step, counts


def _psy_background(t_lo, t_hi, w_hi_gkg, ink):
    """The nomogram behind the data: constant-RH curves from 10 % to saturation.

    Recessive by construction — one thin neutral line per curve, labelled at the
    end where it leaves the frame, so the grid can be read without ever competing
    with the marks on top of it.
    """
    import plotly.graph_objects as go
    grid = "rgba(150,150,144,0.45)"
    traces, labels, placed = [], [], []
    ts = np.linspace(t_lo, t_hi, 160)
    for rh in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        w = _humidity_ratio(ts, rh) * 1000.0
        keep = w <= w_hi_gkg
        if not keep.any():
            continue
        traces.append(go.Scatter(
            x=ts[keep], y=w[keep], mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(color=grid, width=1.6 if rh == 1.0 else 1.0)))
        # Label where the curve LEAVES the frame — right edge for the low-RH
        # curves, top edge for the steep ones — and drop any that would land on
        # top of the previous label, which is what turns the steep end of the
        # family into an unreadable pile.
        x, y = float(ts[keep][-1]), float(w[keep][-1])
        exits_top = y > w_hi_gkg - 0.5
        if any(abs(x - px) < 1.6 and abs(y - py) < 2.0 for px, py in placed):
            continue
        placed.append((x, y))
        labels.append(dict(x=x, y=y, text=f"{int(rh * 100)}%", showarrow=False,
                           xanchor="right" if exits_top else "left",
                           yanchor="top" if exits_top else "bottom",
                           font=dict(size=10, color=ink["muted"])))
    return traces, labels


def _fig_psychrometric(series, mode="light", cat="II", t_step=0.2):
    """Indoor comfort as a PSYCHROMETRIC cloud — operative temperature against
    humidity ratio — one selectable season at a time, one colour per occupant.

    A temperature alone cannot say whether a dwelling was comfortable: 26 °C at
    8 g/kg and 26 °C at 16 g/kg are different rooms to be in. The chart puts the
    hours where they actually fell, over the two comfort criteria the KPI table
    reports — so "how much of this occupant's year sat outside, and in which
    direction" is one glance rather than a table lookup.

    * x — OPERATIVE temperature, not dry-bulb: it is what both criteria are
      defined on (and what ASHRAE 55's graphical method plots), and it carries the
      radiant half of comfort that a dry-bulb chart drops.
    * y — humidity ratio, so the constant-RH curves are the background nomogram
      and equal vertical distances mean equal water.
    * marks — hours binned onto a ``t_step`` grid, opacity by hours in the cell;
      colour is the scenario's, and stays the scenario's whatever is filtered.
    * the filled POLYGON is the extended comfort zone (``comfort_zone`` in the
      config — the Mollier h-x comfort field of EN ISO 7730 / DIN 1946-2 /
      ASHRAE 55): temperature, relative humidity, absolute humidity and enthalpy
      bounds intersected, which is why its top edge bends. Season-independent, as
      the standards define it, and the criterion the ``% in zone`` in each legend
      entry counts against — hour by hour, the same arithmetic as the
      ``pct_hours_out_of_comfort_zone_*`` columns.
    * the dashed OUTLINE is the EN 16798-1 adaptive band at the SELECTED SEASON'S
      MEAN running-mean outdoor temperature — a temperature-only criterion, drawn
      as an outline precisely because it says nothing about the vertical axis.
      This is what moves when the season buttons are used.

    Two criteria rather than one because they disagree, and the disagreement is
    the finding: a dwelling can sit inside the climate-relative adaptive band and
    still be outside the comfort field on humidity.

    Only OCCUPIED hours are plotted, as with every other comfort figure on the
    page. Returns ``None`` when the run carries no humidity output (it was added
    after some runs were made — re-run to get the chart).
    """
    import plotly.graph_objects as go
    from hypercadaster_ES.energy import (_ADAPTIVE_CAT_BAND,
                                         _hourly_season_mask,
                                         _outside_comfort_zone,
                                         _running_mean_outdoor,
                                         comfort_zone_polygon)
    scenarios = _ordered(list(series), _SCENARIO_ORDER)
    if not scenarios:
        return None
    ref = series[scenarios[0]]
    if "w_kgkg" not in ref or not np.isfinite(ref["w_kgkg"].to_numpy(float)).any():
        return None
    ink, colors = _INK[mode], _scenario_colors(scenarios, mode)
    band = _ADAPTIVE_CAT_BAND.get(cat, 3.0)

    # Seasons and the running mean come off the WEATHER, so they are one
    # definition shared by every scenario — the same one the KPI rows use.
    masks, _days, _ranges = _hourly_season_mask(ref, "t_outdoor")
    theta = _running_mean_outdoor(ref, "t_outdoor")
    options = [("whole year", np.ones(len(ref), bool))]
    options += [(s, masks[s]) for s in _SEASON_ORDER if masks.get(s) is not None
                and masks[s].any()]

    # One pass to fix the frame, so every season is read on the same axes and the
    # clouds can be compared between selections rather than only within one.
    t_all, w_all = [], []
    for s in scenarios:
        d = series[s]
        occ = d["occupants"].to_numpy(float) > 0 if "occupants" in d else np.ones(len(d), bool)
        t = d["t_indoor"].to_numpy(float)
        w = d["w_kgkg"].to_numpy(float) * 1000.0
        ok = occ & np.isfinite(t) & np.isfinite(w)
        t_all.append(t[ok])
        w_all.append(w[ok])
    t_all, w_all = np.concatenate(t_all), np.concatenate(w_all)
    if not len(t_all):
        return None
    t_lo = float(np.floor(min(t_all.min(), 14.0) - 1))
    t_hi = float(np.ceil(max(t_all.max(), 30.0) + 1))
    w_hi = float(np.ceil(max(w_all.max() * 1.15, 14.0)))

    fig = go.Figure()
    bg, labels = _psy_background(t_lo, t_hi, w_hi, ink)
    for tr in bg:
        fig.add_trace(tr)
    n_fixed = len(bg)
    # The EXTENDED comfort zone: a closed polygon, traced up its lower edge and
    # back down its upper one. Season-independent — the standards define it on the
    # indoor state alone — so it is drawn once, outside the season blocks.
    zt, zlo, zhi = comfort_zone_polygon()
    if len(zt):
        fig.add_trace(go.Scatter(
            x=np.round(np.concatenate([zt, zt[::-1], zt[:1]]), 2),
            y=np.round(np.concatenate([zlo, zhi[::-1], zlo[:1]]) * 1000.0, 3),
            mode="lines", fill="toself", fillcolor="rgba(150,150,144,0.16)",
            line=dict(color="rgba(120,120,114,0.85)", width=1.2),
            hoverinfo="skip",
            name=f"comfort zone · {zt.min():.0f}–{zt.max():.0f} °C, "
                 f"{zlo.min() * 1000:.1f}–{zhi.max() * 1000:.1f} g/kg"))
        n_fixed += 1

    # Per option: the adaptive band outline, then one cloud per scenario.
    blocks = []
    for label, mask in options:
        start = len(fig.data)
        tc = 0.33 * float(np.nanmean(theta[mask])) + 18.8
        fig.add_trace(go.Scatter(
            x=[tc - band, tc + band, tc + band, tc - band, tc - band],
            y=[0, 0, w_hi, w_hi, 0], mode="lines",
            line=dict(color=ink["muted"], width=1.4, dash="dash"),
            hoverinfo="skip", visible=False,
            name=f"EN 16798-1 cat {cat} adaptive band · "
                 f"{tc - band:.1f}–{tc + band:.1f} °C"))
        for s in scenarios:
            d = series[s]
            occ = (d["occupants"].to_numpy(float) > 0
                   if "occupants" in d else np.ones(len(d), bool))
            t = d["t_indoor"].to_numpy(float)
            w = d["w_kgkg"].to_numpy(float) * 1000.0
            ok = mask & occ & np.isfinite(t) & np.isfinite(w)
            if not ok.any():
                continue
            # The headline is the EXTENDED zone — the criterion the polygon draws
            # and the one that uses both axes of the chart it is printed on.
            in_zone = ~_outside_comfort_zone(t[ok], w[ok] / 1000.0)
            bt, bw, hours = _bin_psy_points(t[ok], w[ok], t_step, t_step)
            peak = max(float(np.percentile(hours, 95)), 1.0)
            # Rounded before they are serialised: a bin centre is a multiple of
            # t_step, but 24.400000000000002 costs 18 characters of JSON in a page
            # that carries tens of thousands of them.
            bt, bw = np.round(bt, 3), np.round(bw, 3)
            alpha = np.round(0.18 + 0.72 * np.clip(hours / peak, 0.0, 1.0), 3)
            rh = np.round(_relative_humidity(bt, bw / 1000.0) * 100.0, 1)
            fig.add_trace(go.Scatter(
                x=bt, y=bw, mode="markers", visible=False, legendgroup=s,
                name=f"{s} · {100.0 * in_zone.mean():.0f}% in zone",
                marker=dict(color=colors[s], size=5, opacity=alpha,
                            line=dict(width=0)),
                customdata=np.stack([rh, hours], axis=1),
                hovertemplate=(f"<b>{s}</b><br>%{{x:.1f}} °C operative<br>"
                               "%{y:.1f} g/kg · %{customdata[0]:.0f} % RH<br>"
                               "%{customdata[1]:.0f} occupied hours"
                               "<extra></extra>")))
        blocks.append((label, start, len(fig.data)))

    for i, (_label, start, end) in enumerate(blocks):
        if i == 0:
            for j in range(start, end):
                fig.data[j].visible = True

    def _vis(sel):
        out = [True] * n_fixed
        for k, (_label, start, end) in enumerate(blocks):
            out += [k == sel] * (end - start)
        return out

    fig = _fig_shell(fig, mode, height=560,
                     xlab="operative temperature (°C)",
                     ylab="humidity ratio (g water / kg dry air)")
    fig.update_layout(
        annotations=labels,
        # The season filter sits in one row above the plot, where a filter belongs.
        updatemenus=[dict(
            type="buttons", direction="right", showactive=True, x=0, xanchor="left",
            y=1.13, yanchor="top", pad=dict(t=0, b=0),
            bgcolor="rgba(0,0,0,0)", bordercolor=ink["grid"],
            font=dict(color=ink["secondary"], size=12),
            buttons=[dict(label=label, method="update",
                          args=[{"visible": _vis(i)}])
                     for i, (label, _s, _e) in enumerate(blocks)])])
    fig.update_xaxes(range=[t_lo, t_hi])
    fig.update_yaxes(range=[0, w_hi], gridcolor="rgba(0,0,0,0)")
    return fig


def _fig_air_hygiene(hourly, mode="light", year=None, window=None, daily=True):
    """The three quantities that together say whether the air is healthy: the CO2
    the occupants breathe, the outdoor air renewing it, and the excess over the
    outdoor background that the ventilation has to remove.

    Three panels on a shared time axis rather than three scales on one plot: they
    are measured in different units and a common axis would flatten two of them.
    ``daily=True`` aggregates to daily means; otherwise the hours are drawn as
    they came out of the simulation.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    scenarios = _ordered(list(hourly), _SCENARIO_ORDER)
    if not scenarios:
        return None
    colors = _scenario_colors(scenarios, mode)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.4, 0.3, 0.3])
    panels = (("co2_ppm", "CO₂ ppm", "{:.0f} ppm"),
              ("ach_total", "air change ACH", "{:.2f} ACH"),
              ("co2_excess_ppm", "CO₂ generated ppm", "{:.0f} ppm"))
    drew = False
    for s in scenarios:
        h = hourly[s]
        if window is not None:
            keep = pd.Series([(m, d) in window for m, d in zip(h["month"], h["day"])],
                             index=h.index)
            h = h[keep]
        if h.empty:
            continue
        d = _daily(h) if daily else h
        if daily:
            x = [_weekday_stamp(m, dd, 0, year).replace(" 00h", "")
                 for m, dd in zip(d["month"], d["day"])]
        else:
            x = [_weekday_stamp(m, dd, hr, year)
                 for m, dd, hr in zip(d["month"], d["day"], d["hour"])]
        for r, (col, _lab, tpl) in enumerate(panels, start=1):
            if col not in d:
                continue
            fig.add_trace(go.Scatter(
                x=x, y=d[col], mode="lines", name=s, legendgroup=s,
                showlegend=(r == 1),
                line=dict(color=colors[s], width=2, dash=_scenario_dash(s)),
                hovertemplate=f"<b>{s}</b><br>%{{x}}<br>{tpl.replace('{:.0f}', '%{y:.0f}').replace('{:.2f}', '%{y:.2f}')}<extra></extra>"),
                row=r, col=1)
            drew = True
    if not drew:
        return None
    fig = _fig_shell(fig, mode, height=700)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(type="category", nticks=10, tickangle=0)
    for r, (_col, lab, _t) in enumerate(panels, start=1):
        fig.update_yaxes(title_text=lab, row=r, col=1)
    return fig


def _fig_week_detail(hourly, kind="cooling", mode="light", days=7):
    """The extreme week at HOURLY resolution: indoor temperature above, HVAC
    demand below, sharing the x-axis.

    Two stacked panels rather than one chart with a second y-scale. A dual axis
    lets the reader's eye read a crossing that the data does not contain — the
    apparent relationship changes with whichever scaling was picked — so the two
    measures get their own panel and the shared x does the pairing.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    scenarios = _ordered(list(hourly), _SCENARIO_ORDER)
    if not scenarios:
        return None
    colors = _scenario_colors(scenarios, mode)
    ink = _INK[mode]
    window = _worst_week(_daily(hourly[scenarios[0]]), kind, days)
    if not window:
        return None
    # Axis titles on the LEFT rather than centred panel titles: a centred title
    # sits exactly where Plotly floats its modebar.
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.58, 0.42])
    first = True
    for s in scenarios:
        h = hourly[s]
        keep = pd.Series([(m, d) in window for m, d in zip(h["month"], h["day"])],
                         index=h.index)
        sub = h[keep]
        if sub.empty:
            continue
        x = [f"{int(m):02d}-{int(d):02d} {int(hr):02d}h"
             for m, d, hr in zip(sub["month"], sub["day"], sub["hour"])]
        if first:
            fig.add_trace(go.Scatter(
                x=x, y=sub["t_outdoor"], mode="lines", name="outdoor",
                line=dict(color=ink["muted"], width=1.5, dash="dot"),
                hovertemplate="outdoor %{y:.1f} °C<extra></extra>"), row=1, col=1)
            first = False
        fig.add_trace(go.Scatter(
            x=x, y=sub["t_indoor"], mode="lines", name=s, legendgroup=s,
            line=dict(color=colors[s], width=2, dash=_scenario_dash(s)),
            hovertemplate=f"<b>{s}</b><br>%{{x}}<br>%{{y:.1f}} °C<extra></extra>"),
            row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=sub["hvac_kwh"], mode="lines", name=s, legendgroup=s,
            showlegend=False,
            line=dict(color=colors[s], width=2, dash=_scenario_dash(s)),
            hovertemplate=f"<b>{s}</b><br>%{{x}}<br>%{{y:.2f}} kWh<extra></extra>"),
            row=2, col=1)
    fig = _fig_shell(fig, mode, height=560)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(type="category", nticks=8, tickangle=0)
    fig.update_yaxes(title_text="indoor °C", row=1, col=1)
    fig.update_yaxes(title_text="HVAC kWh/h", row=2, col=1)
    return fig


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
_CSS = """
:root { color-scheme: light dark; }
.viz-root {
  --surface-0: #ffffff; --surface-1: #fcfcfb; --surface-2: #f4f3f0;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #6f6e6a;
  --border: #e4e3df;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    --surface-0: #121211; --surface-1: #1a1a19; --surface-2: #232322;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #93928a;
    --border: #333230;
  }
}
:root[data-theme="dark"] .viz-root {
  --surface-0: #121211; --surface-1: #1a1a19; --surface-2: #232322;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #93928a;
  --border: #333230;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); }
.viz-root {
  background: var(--surface-0); color: var(--text-primary);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
        Helvetica, Arial, sans-serif;
  padding: 32px 28px 64px; max-width: 1180px; margin: 0 auto;
}
h1 { font-size: 24px; font-weight: 650; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 17px; font-weight: 620; margin: 40px 0 4px;
     padding-top: 20px; border-top: 1px solid var(--border); }
h3 { font-size: 14px; font-weight: 600; margin: 24px 0 2px;
     color: var(--text-secondary); }
p.sub { color: var(--text-secondary); margin: 0 0 8px; font-size: 13.5px; }
p.note { color: var(--text-muted); margin: 2px 0 14px; font-size: 12.5px;
         max-width: 78ch; }
.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin: 20px 0 8px; }
.tile { background: var(--surface-1); border: 1px solid var(--border);
        border-radius: 10px; padding: 12px 16px; min-width: 148px; flex: 1 1 148px; }
.tile .k { font-size: 11.5px; color: var(--text-muted); text-transform: uppercase;
           letter-spacing: 0.04em; }
.tile .v { font-size: 22px; font-weight: 640; margin-top: 3px;
           font-variant-numeric: tabular-nums; }
.tile .u { font-size: 12px; color: var(--text-muted); font-weight: 400; }
.figure { background: var(--surface-1); border: 1px solid var(--border);
          border-radius: 10px; padding: 8px 8px 4px; margin: 10px 0 4px; }
.tablewrap { overflow-x: auto; border: 1px solid var(--border);
             border-radius: 10px; background: var(--surface-1); }
table { border-collapse: collapse; width: 100%; font-size: 12.5px;
        font-variant-numeric: tabular-nums; }
th, td { padding: 7px 10px; text-align: right; white-space: nowrap;
         border-bottom: 1px solid var(--border); }
th { position: sticky; top: 0; background: var(--surface-2);
     color: var(--text-secondary); font-weight: 600; text-align: right; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2),
th:nth-child(3), td:nth-child(3) { text-align: left; }
tbody tr:last-child td { border-bottom: none; }
.swatch { display: inline-block; width: 9px; height: 9px; border-radius: 2px;
          margin-right: 7px; vertical-align: baseline; }
.foot { color: var(--text-muted); font-size: 12px; margin-top: 34px; }
"""

_THEME_JS = """
(function () {
  // Plotly bakes colours into each figure, so the ink is restyled when the page
  // resolves to dark. The series hues are re-stepped for the dark surface rather
  // than inverted, matching how the palette is defined.
  var LIGHT = %LIGHT%, DARK = %DARK%;
  var INK_L = %INK_L%, INK_D = %INK_D%;
  function isDark() {
    var t = document.documentElement.getAttribute('data-theme');
    if (t === 'dark') return true;
    if (t === 'light') return false;
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function apply() {
    if (!window.Plotly) return;
    var dark = isDark(), ink = dark ? INK_D : INK_L;
    var map = {};
    for (var i = 0; i < LIGHT.length; i++) map[LIGHT[i].toLowerCase()] = DARK[i];
    var inv = {};
    for (var j = 0; j < DARK.length; j++) inv[DARK[j].toLowerCase()] = LIGHT[j];
    document.querySelectorAll('.js-plotly-plot').forEach(function (gd) {
      (gd.data || []).forEach(function (tr) {
        var m = tr.marker || {}, l = tr.line || {};
        [[m, 'color'], [l, 'color']].forEach(function (p) {
          var c = (p[0][p[1]] || '').toString().toLowerCase();
          if (dark && map[c]) p[0][p[1]] = map[c];
          else if (!dark && inv[c]) p[0][p[1]] = inv[c];
        });
      });
      Plotly.relayout(gd, {
        'font.color': ink.secondary, 'legend.font.color': ink.secondary,
        'xaxis.linecolor': ink.grid, 'xaxis.tickcolor': ink.grid,
        'xaxis.tickfont.color': ink.secondary, 'xaxis.title.font.color': ink.muted,
        'yaxis.gridcolor': ink.grid, 'yaxis.tickfont.color': ink.secondary,
        'yaxis.title.font.color': ink.muted
      });
      Plotly.redraw(gd);
    });
  }
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', apply);
  new MutationObserver(apply).observe(document.documentElement,
      { attributes: true, attributeFilter: ['data-theme'] });
  if (document.readyState !== 'loading') apply();
  else document.addEventListener('DOMContentLoaded', apply);
})();
"""


def _tiles(res, use="residential"):
    """Headline numbers for ONE use: the spread across occupants is the story, so
    the tiles carry the range rather than one scenario's value."""
    conditioned = res[~res["scenario"].astype(str).str.endswith("_ff")]
    out = []

    def _tile(k, v, u=""):
        out.append(f'<div class="tile"><div class="k">{html.escape(k)}</div>'
                   f'<div class="v">{v}<span class="u"> {html.escape(u)}</span>'
                   f'</div></div>')

    _tile("scenarios", f"{res['scenario'].nunique()}")
    if "floor_area_m2" in res.columns and len(res):
        _tile(f"{use} area", f"{res['floor_area_m2'].max():,.0f}", "m²")
    if len(conditioned) and "thermal_demand_kwh_m2" in conditioned.columns:
        by = conditioned.groupby("scenario")["thermal_demand_kwh_m2"].sum()
        if len(by):
            _tile("lowest demand", f"{by.min():.1f}", f"kWh/m² · {by.idxmin()}")
            _tile("highest demand", f"{by.max():.1f}", f"kWh/m² · {by.idxmax()}")
            if by.min() > 0:
                _tile("spread", f"{by.max() / by.min():.1f}×", "between occupants")
    if "n_dwellings" in res.columns and res["n_dwellings"].notna().any():
        _tile("dwellings", f"{int(res['n_dwellings'].dropna().iloc[0])}")
    if "people_per_dwelling" in res.columns and res["people_per_dwelling"].notna().any():
        _tile("people per dwelling", f"{res['people_per_dwelling'].dropna().iloc[0]:.1f}")
    return "".join(out)


def _table(df, colors=None):
    """The KPI frame as a table — the accessible view of every chart above, and
    the relief the palette's contrast warning requires."""
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                txt = "" if v != v else f"{v:,.2f}".rstrip("0").rstrip(".")
            else:
                txt = "" if v is None else str(v)
            if c == "scenario" and colors and v in colors:
                txt = (f'<span class="swatch" style="background:{colors[v]}"></span>'
                       f'{html.escape(txt)}')
            else:
                txt = html.escape(txt)
            cells.append(f"<td>{txt}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def scenario_report(kpis, output_path_html=None, output_dir=None,
                    building_reference=None, title=None, zone_areas=None,
                    detail_days=7, output_path_csv=None, output_path=None,
                    year=None, zone_uses=None):
    """Write a self-contained HTML report of an occupant-behaviour simulation —
    **one page per USE**.

    A building is not one thing. Averaging a dwelling with the parking box below it
    and the shop beside it produces a page that describes neither: the parking lifts
    the air-change curve and flattens the CO2 towards the outdoor background, the
    shop's opening hours cut across the household's agenda. So every use present in
    ``kpis`` gets its own page, over its own zones, named
    ``{output_path_html}_{use}.html``.

    Two halves on each page. The **summary** compares what the scenarios cost and
    deliver — thermal demand, gains, delivered setpoint, discomfort, the
    indoor-temperature spread and DHW — from that use's KPI rows. The **raw
    outputs** show the hourly EnergyPlus series those rows were summarised from,
    over that use's zones only.

    Plotly is inlined, so the page opens from disk with no network.

    Parameters
    ----------
    kpis : pandas.DataFrame
        What :func:`hypercadaster_ES.energy.run_behavior_scenarios` returned.
    output_path_html : str, optional
        Base path of the reports. Each page is written to ``{stem}_{use}.html``,
        so one call can leave ``…_scenarios_residential.html`` beside
        ``…_scenarios_commercial.html``.
    output_path_csv : str, optional
        Where to also write ``kpis`` as CSV, so the numbers behind the page are
        available to a spreadsheet or the next script without re-running anything.
        Written even when no HTML is requested.
    output_path : str, optional
        Deprecated alias for ``output_path_html``.
    output_dir : str, optional
        The scenario runner's ``output_dir``, holding the per-scenario
        ``{building}_{scenario}/eplusout.csv``. Given it, the report adds the raw
        hourly section; without it, only the summary is written.
    building_reference : str, optional
        Used to strip the prefix off the run directories and to title the page.
    zone_areas : dict, optional
        ``{ZONE_NAME: floor area}`` (``energy._zone_floor_areas(idf)``) so the
        building indoor temperature is floor-area weighted rather than a plain
        mean of unequal rooms.
    zone_uses : dict, optional
        ``{ZONE_NAME: use category}`` (``energy.zone_use_categories(idf)``), which
        is what splits the hourly series by use. Omitted, it is read from the
        builder's ``.hcmeta.json`` sidecar beside the IDF in ``output_dir`` (or its
        parent); if that is missing too, every page falls back to showing all
        zones.
    detail_days : int, default 7
        Length of the hourly detail window.
    year : int, optional
        Calendar year of the simulated weather. Only used to label the hourly
        charts with the weekday, which is what makes a weekly occupancy rhythm
        legible — a bare date hides it.

    Returns
    -------
    dict
        ``{use category: HTML path}`` — one page per use — or ``{}`` when only a
        CSV was requested.
    """
    kpis = kpis.copy()
    if "scenario" not in kpis.columns:
        raise ValueError("kpis has no 'scenario' column — pass the frame "
                         "run_behavior_scenarios returned")

    def _prepare_path(path, ext):
        if not str(path).lower().endswith(ext):
            path = f"{path}{ext}"
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        return path

    if output_path_csv:
        kpis.to_csv(_prepare_path(output_path_csv, ".csv"), index=False)
    output_path_html = output_path_html or output_path
    if not output_path_html:
        return {}

    hourly = _read_scenario_hourly(output_dir, building_reference)
    zone_uses = _resolve_zone_uses(zone_uses, output_dir, building_reference)
    out = {}
    for use in _report_uses(kpis):
        sub = (kpis[kpis["use_type"].astype(str).str.lower().str.startswith(use)]
               if "use_type" in kpis.columns else kpis)
        if not len(sub):
            continue
        zones = {z for z, u in zone_uses.items() if u == use} or None
        doc = _use_document(sub, use, hourly=hourly, zones=zones,
                            zone_areas=zone_areas, building_reference=building_reference,
                            title=title, detail_days=detail_days, year=year)
        path = _prepare_path(_use_output_path(output_path_html, use), ".html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
        out[use] = path
    return out


def _report_uses(kpis):
    """The use categories a report has to be written for, in a stable order —
    residential first, because it is the one the occupant scenarios are about."""
    if "use_type" not in kpis.columns:
        return ["residential"]
    seen = {str(u).strip().lower() for u in kpis["use_type"].dropna().unique()}
    order = [u for u in ("residential", "commercial", "office") if u in seen]
    return order + sorted(seen - set(order))


def _use_output_path(path, use):
    """``…/X_scenarios.html`` → ``…/X_scenarios_residential.html``. Every page is
    suffixed, including the single-use case, so a script can build the name it
    wants without first knowing which uses the building turned out to have."""
    stem = str(path)
    if stem.lower().endswith(".html"):
        stem = stem[:-5]
    return f"{stem}_{use}"


def _resolve_zone_uses(zone_uses, output_dir, building_reference):
    """``{ZONE: use category}``, from the caller or from the ``.hcmeta.json``
    sidecar the builder wrote beside its IDF.

    Without it the report cannot tell a dwelling from the parking box below it, and
    falls back to reporting every zone on every page — which is what it did before
    the per-use split, and is still better than reporting nothing.
    """
    if zone_uses:
        return {str(z).upper(): str(u).strip().lower() for z, u in zone_uses.items()}
    if not output_dir:
        return {}
    from hypercadaster_ES.energy import _read_idf_metadata, _use_category
    pattern = f"{building_reference}*.idf" if building_reference else "*.idf"
    for d in (output_dir, os.path.dirname(os.path.abspath(output_dir))):
        for idf_path in sorted(glob.glob(os.path.join(d, pattern))):
            meta = _read_idf_metadata(idf_path) or {}
            zm = meta.get("zone_meta") or {}
            if zm:
                return {str(z).upper(): _use_category((s or {}).get("use_type"))
                        for z, s in zm.items()}
    return {}


def _use_document(kpis, use, hourly=None, zones=None, zone_areas=None,
                  building_reference=None, title=None, detail_days=7, year=None):
    """The HTML of ONE use's report — the summary comparisons over that use's KPI
    rows, and the raw hourly series over that use's zones only."""
    import plotly.io as pio

    hourly = hourly or {}
    scenarios = _ordered(kpis["scenario"].unique(), _SCENARIO_ORDER)
    colors = _scenario_colors(scenarios, "light")
    residential = kpis

    parts = []
    figs = []

    def _add_fig(fig, heading=None, note=None, modebar=True):
        if fig is None:
            return
        first = not figs
        figs.append(fig)
        block = pio.to_html(fig, full_html=False,
                            include_plotlyjs="inline" if first else False,
                            config={"displaylogo": False, "responsive": True,
                                    "displayModeBar": modebar,
                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
        if heading:
            parts.append(f"<h3>{html.escape(heading)}</h3>")
        if note:
            parts.append(f'<p class="note">{html.escape(note)}</p>')
        parts.append(f'<div class="figure">{block}</div>')

    name = building_reference or "building"
    head_title = title or f"Occupant-behaviour scenarios — {name}"
    parts.append(f"<h1>{html.escape(head_title)} · {html.escape(use)}</h1>")
    parts.append('<p class="sub">每 scenario is the same building under a different '
                 'occupant; the fabric, geometry and weather are identical. This page '
                 f'covers the {html.escape(use)} zones ONLY — every other use has its '
                 'own, because averaging them describes neither.</p>'
                 .replace("每", "Every"))
    parts.append(f'<div class="tiles">{_tiles(kpis, use)}</div>')

    # ---- Summary comparisons ------------------------------------------------
    # Numbered so the page has a table of contents you can point at, and grouped
    # so related figures share a number: everything about outdoor air is 7,
    # everything about a distribution is 8, everything about CO2 is 9.
    parts.append("<h2>How the occupants compare</h2>")
    if len(residential):
        _add_fig(_bar_by_scenario_season(residential, "thermal_demand_kwh_m2", "kWh/m²"),
                 "1. Thermal demand", f"HVAC delivered per m² of {use} floor "
                 "area. The free-floating (_ff) runs have no HVAC, so they read "
                 "zero by construction — they are there for the temperatures.")
        for num, col, head, note in (
                ("2", "solar_gains_kwh_m2", "Solar gains",
                 "Transmitted through the glazing. The gap between occupants in "
                 "the cooling season is the persianas and awnings being operated "
                 "— or not."),
                ("3", "internal_gains_kwh_m2", "Internal gains",
                 "People, lights and equipment, driven by the census occupancy."),
                ("4", "avg_setpoint_c", "Delivered setpoints",
                 "Averaged over the hours the HVAC actually ran, so setback and "
                 "free-floating hours do not dilute it."),
                ("5", "dhw_demand_kwh_m2", "Domestic hot water",
                 "From the census-tract occupancy, the same head count every "
                 "scenario is given.")):
            if col in residential.columns and residential[col].notna().any():
                _add_fig(_bar_by_scenario_season(
                    residential, col, "°C" if col.endswith("_c") else "kWh/m²"),
                    f"{num}. {head}", note)
        if "pct_hours_out_of_comfort_day" in residential.columns:
            _add_fig(_bar_by_scenario_season(residential,
                                             "pct_hours_out_of_comfort_day", "%"),
                     "6.1. Thermal discomfort, day (08:00–21:59)",
                     "EN 16798-1 adaptive, over OCCUPIED hours only.")
            _add_fig(_bar_by_scenario_season(residential,
                                             "pct_hours_out_of_comfort_night", "%"),
                     "6.2. Thermal discomfort, night (22:00–07:59)", None)
        if ("pct_hours_out_of_comfort_zone_day" in residential.columns
                and residential["pct_hours_out_of_comfort_zone_day"].notna().any()):
            _add_fig(_bar_by_scenario_season(
                residential, "pct_hours_out_of_comfort_zone_day", "%"),
                "6.3. Outside the extended comfort zone, day",
                "The same occupied hours against the Mollier h-x comfort field "
                "(EN ISO 7730 / DIN 1946-2 / ASHRAE 55) rather than a temperature "
                "band: 20–26 °C AND 30–65 % RH AND under the absolute-humidity and "
                "enthalpy limits. An hour can pass 6.1 and fail this one on "
                "humidity alone — figure 2 shows where.")
            _add_fig(_bar_by_scenario_season(
                residential, "pct_hours_out_of_comfort_zone_night", "%"),
                "6.4. Outside the extended comfort zone, night", None)
        # 7 — outdoor air, split by route and by time of day. Infiltration is the
        # fabric and identical across occupants; only the window column is
        # behaviour, which is why they are shown apart.
        for num, col, head, note in (
                ("7.1", "ach_infiltration_mean_day",
                 "Outdoor-air infiltration rate, mean while occupied, day",
                 "The fabric's own leakage — the same for every occupant of the "
                 "same building, which is what makes it the baseline the window "
                 "column is read against."),
                ("7.2", "ach_ventilation_mean_day",
                 "Outdoor-air window ventilation rate, day",
                 "The part the occupant controls."),
                ("7.3", "ach_infiltration_mean_night",
                 "Outdoor-air infiltration rate, mean while occupied, night", None),
                ("7.4", "ach_ventilation_mean_night",
                 "Outdoor-air window ventilation rate, night",
                 "Night is when the dwelling is full and the windows are usually "
                 "shut, so this column and the CO₂ below move together.")):
            if col in residential.columns and residential[col].notna().any():
                _add_fig(_bar_by_scenario_season(residential, col, "ACH",
                                                 fmt="{:.2f}"),
                         f"{num}. {head}", note)
        # 8 — the distributions. A mean hides the shape of both of these.
        if ("ach_total_mean_day" in residential.columns
                and residential["ach_total_mean_day"].notna().any()):
            _add_fig(_fig_distribution_bands(residential, prefix="ach_total",
                                             unit=" ACH", fmt="{:.2f}",
                                             center="mean"),
                     "8.1. Outdoor-air change rate spread",
                     "Total outdoor-air change over occupied hours: the hairline "
                     "spans min–max, the medium band p10–p90, the thick band "
                     "p20–p80, and the marker is the MEAN — the air actually "
                     "renewed. No median is reported on any of the outdoor-air "
                     "rates: a dwelling opened for one hour in twenty-one has a "
                     "median of zero, which is not what it breathed. The spread "
                     "is what separates it from one left permanently ajar at the "
                     "same mean.", modebar=False)
        if any(c.startswith("t_p50_") for c in residential.columns):
            _add_fig(_fig_distribution_bands(residential, prefix="t", unit=" °C"),
                     "8.2. Indoor temperature spread",
                     "Same construction, on the operative temperature. Day is "
                     "solid, night faded.", modebar=False)
        # 9 — CO2, the air-quality answer to the outdoor air above.
        for num, col, head, note in (
                ("9.1", "co2_mean_day", "Indoor CO₂, mean while occupied, day",
                 "How renewed the air is: the occupants raise it, leakage and "
                 "open windows flush it. Outdoor background is ~420 ppm. Read the "
                 "normative case apart from the occupant ones: it keeps the "
                 "standard's fixed occupancy profile and its waking metabolic "
                 "rate around the clock, so its CO₂ answers to the law's user, "
                 "not to a measured day."),
                ("9.2", "co2_mean_night", "Indoor CO₂, mean while occupied, night",
                 "Night is where CO₂ bites: the dwelling is full, everyone is "
                 "asleep, and the windows are usually shut.")):
            if col in residential.columns and residential[col].notna().any():
                _add_fig(_bar_by_scenario_season(residential, col, "ppm",
                                                 fmt="{:.0f}"),
                         f"{num}. {head}", note)

    # ---- Raw hourly ---------------------------------------------------------
    # Restricted to this use's zones: a parking box with no occupants and a shop
    # open half the day answer to nothing the dwelling charts above are asking.
    if hourly:
        parts.append("<h2>Raw simulation output</h2>")
        parts.append('<p class="sub">The hourly EnergyPlus series the rows above '
                     f'were summarised from, over the {html.escape(use)} zones.</p>')
        # KPI scenario label → the run directory it was summarised from. They are
        # the same everywhere except the non-residential zones, which carry ONE
        # normative user in every run and are therefore reported once, out of the
        # run the KPI builder took them from.
        runs = {s: s for s in scenarios if s in hourly}
        for label, source in _nonresidential_run().items():
            if label in scenarios and source in hourly:
                runs[label] = source
        series = {label: _hourly_building_series(hourly[key], zone_areas, zones)
                  for label, key in runs.items()}
        if not series:
            series = {k: _hourly_building_series(v, zone_areas, zones)
                      for k, v in hourly.items()}
        _add_fig(_fig_daily_series(series, "t_indoor", "°C", with_outdoor=True),
                 "1. Indoor temperature, daily mean",
                 f"Floor-area weighted across the {use} zones, against the "
                 "outdoor dry bulb.")
        _add_fig(_fig_psychrometric(series),
                 "2. Thermal comfort, psychrometric",
                 "Every OCCUPIED hour as a point in (operative temperature, "
                 "humidity ratio), one colour per occupant, over BOTH comfort "
                 "criteria. The filled polygon is the extended comfort zone — the "
                 "Mollier h-x comfort field of EN ISO 7730 / DIN 1946-2 / "
                 "ASHRAE 55, whose top edge bends where the enthalpy limit takes "
                 "over from the relative-humidity ceiling; it is what the "
                 "'% in zone' beside each occupant is counted against, hour by "
                 "hour. The dashed outline is the EN 16798-1 adaptive band, a "
                 "temperature-only criterion, which is why it is an outline and "
                 "why it is the thing that moves when you pick a season. Marker "
                 "opacity is hours per cell, so the dense core of a year is "
                 "visible through its outliers.")
        _add_fig(_fig_daily_series(series, "hvac_kwh", "kWh/day"),
                 "3. HVAC energy, daily total", None)
        _add_fig(_fig_daily_series(series, "solar_kwh", "kWh/day"),
                 "4. Solar gains, daily total", None)
        _add_fig(_fig_weekly_profile(series, "occupants", "people", year=year,
                                     fmt="{:.2f}"),
                 "5. Occupancy of people, average week",
                 "People in the building, summed across its zones, averaged over "
                 "each hour of the WEEK — the driver behind the internal gains, "
                 "the hot water and the CO₂. Occupancy is a weekly rhythm, so a "
                 "daily mean flattened the one thing worth seeing: who is home at "
                 "03:00, who leaves at nine, and how Saturday differs from "
                 "Tuesday.")
        _add_fig(_fig_air_hygiene(series, year=year, daily=True),
                 "6.1. Air hygiene conditions, daily mean",
                 "CO₂ breathed, the outdoor-air change renewing it, and the "
                 "excess over the ~420 ppm background that the ventilation has "
                 "to remove.")
        _add_fig(_fig_air_hygiene(series, year=year, daily=False,
                                  window=_worst_co2_week(
                                      _daily(series[_ordered(list(series),
                                                             _SCENARIO_ORDER)[0]]),
                                      detail_days)),
                 f"6.2. Air hygiene conditions — the {detail_days} worst days",
                 "The same three, hourly. A daily mean cannot show whether the "
                 "air recovers between occupied periods: an overnight build-up "
                 "followed by a morning airing averages to the same number as a "
                 "flat mediocre day.")
        ref_daily = _daily(series[_ordered(list(series), _SCENARIO_ORDER)[0]])
        for i, (kind, word) in enumerate((("cooling", "hottest"),
                                          ("heating", "coldest")), start=7):
            _add_fig(_fig_week_detail(series, kind=kind, days=detail_days),
                     f"{i}.1. Worst {kind} week — the {detail_days} {word} days",
                     "Indoor temperature above, HVAC demand below, on a shared "
                     "time axis. Two panels rather than a second y-scale: with "
                     "two scalings on one plot the eye reads crossings the data "
                     "does not contain.")
            _add_fig(_fig_air_hygiene(series, year=year, daily=False,
                                      window=_worst_week(ref_daily, kind,
                                                         detail_days)),
                     f"{i}.2. Air hygiene over the same {kind} week",
                     "Whether the windows opened (or not) for temperature also "
                     "renewed the air.")

    # ---- Table --------------------------------------------------------------
    parts.append("<h2>All KPIs</h2>")
    parts.append('<p class="sub">Every number behind the charts.</p>')
    parts.append(_table(kpis, colors))
    parts.append('<p class="foot">Generated by hypercadaster_ES · '
                 'energy.run_behavior_scenarios → reporting.scenario_report</p>')

    js = (_THEME_JS
          .replace("%LIGHT%", str(list(_SERIES_LIGHT)).replace("'", '"'))
          .replace("%DARK%", str(list(_SERIES_DARK)).replace("'", '"'))
          .replace("%INK_L%", str(_INK["light"]).replace("'", '"'))
          .replace("%INK_D%", str(_INK["dark"]).replace("'", '"')))
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(title or f'Scenarios — {name}')} · "
            f"{html.escape(use)}</title>"
            f"<style>{_CSS}</style></head><body>"
            f"<div class='viz-root'>{''.join(parts)}</div>"
            f"<script>{js}</script></body></html>")
