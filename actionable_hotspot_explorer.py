from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


DATA_PATH = Path("shootings_nyc.csv")
OUTPUT_DIR = Path("outputs/spatial")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RADIUS_M = 6_371_008.8
CELL_M = 250
BORO_COLORS = {
    "BRONX": "#00d4ff",
    "BROOKLYN": "#ff5a5f",
    "MANHATTAN": "#ffe66d",
    "QUEENS": "#7bed9f",
    "STATEN ISLAND": "#c56cf0",
}
VIEW_COLORS = {
    "Recent hotspots": "#ff7b72",
    "Emerging hotspots": "#ffa657",
    "Persistent hotspots": "#7ee787",
    "Fading hotspots": "#79c0ff",
}


def load_points():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["OCCUR_DATE"])
    df["year"] = df["date"].dt.year

    # Source coordinate values are swapped in this file.
    df["lon"] = df["Latitude"]
    df["lat"] = df["Longitude"]
    df = df[df["lon"].between(-75, -73) & df["lat"].between(40, 41)].copy()
    df = df[df["BORO"].isin(BORO_COLORS)].copy()

    raw_time = df["OCCUR_TIME"].astype(str)
    hhmmss_hour = pd.to_numeric(raw_time.str.extract(r"^(\d{1,2}):")[0], errors="coerce")
    datetime_hour = pd.to_datetime(raw_time, errors="coerce", format="mixed").dt.hour
    df["hour"] = hhmmss_hour.fillna(datetime_hour).astype(int)
    df["loc_class"] = df["LOC_CLASSFCTN_DESC"].replace({"(null)": pd.NA}).fillna("Unknown")

    lat_ref = np.deg2rad(df["lat"].mean())
    lon_ref = np.deg2rad(df["lon"].mean())
    df.attrs["lat_ref"] = lat_ref
    df.attrs["lon_ref"] = lon_ref
    df["x_m"] = (np.deg2rad(df["lon"]) - lon_ref) * RADIUS_M * np.cos(lat_ref)
    df["y_m"] = (np.deg2rad(df["lat"]) - lat_ref) * RADIUS_M
    df["gx"] = np.floor(df["x_m"] / CELL_M).astype(int)
    df["gy"] = np.floor(df["y_m"] / CELL_M).astype(int)
    return df


def xy_to_lonlat(df, x, y):
    lat_ref = df.attrs["lat_ref"]
    lon_ref = df.attrs["lon_ref"]
    lon = np.rad2deg(x / (RADIUS_M * np.cos(lat_ref)) + lon_ref)
    lat = np.rad2deg(y / RADIUS_M + lat_ref)
    return lon, lat


def build_cells(df):
    early = df[df["year"].between(2006, 2011)]
    recent = df[df["year"].between(2023, 2025)]
    rows = []

    grouped = df.groupby(["BORO", "gx", "gy"])
    for (boro, gx, gy), sub in grouped:
        early_count = len(early[(early["BORO"].eq(boro)) & (early["gx"].eq(gx)) & (early["gy"].eq(gy))])
        recent_count = len(recent[(recent["BORO"].eq(boro)) & (recent["gx"].eq(gx)) & (recent["gy"].eq(gy))])
        total_count = len(sub)
        if max(total_count, recent_count, early_count) < 3:
            continue

        annual_counts = sub.groupby("year").size()
        years_active = int(annual_counts.gt(0).sum())
        years_ge3 = int(annual_counts.ge(3).sum())
        lon, lat = xy_to_lonlat(df, sub["x_m"].mean(), sub["y_m"].mean())
        precinct = int(sub["PRECINCT"].mode().iat[0])
        loc_class = sub["loc_class"].mode().iat[0]
        late_share = sub["hour"].isin([20, 21, 22, 23, 0, 1, 2, 3]).mean() * 100
        street_share = sub["loc_class"].eq("STREET").mean() * 100
        housing_share = sub["loc_class"].eq("HOUSING").mean() * 100

        rows.append(
            {
                "boro": boro,
                "gx": gx,
                "gy": gy,
                "lat": lat,
                "lon": lon,
                "total_count": total_count,
                "early_count": early_count,
                "recent_count": recent_count,
                "early_rate": early_count / 6,
                "recent_rate": recent_count / 3,
                "rate_diff": recent_count / 3 - early_count / 6,
                "years_active": years_active,
                "years_ge3": years_ge3,
                "precinct": precinct,
                "loc_class": loc_class,
                "late_share": late_share,
                "street_share": street_share,
                "housing_share": housing_share,
            }
        )

    cells = pd.DataFrame(rows)
    cells["recent_rank_boro"] = cells.groupby("boro")["recent_count"].rank(method="first", ascending=False)
    cells["emerging_rank_boro"] = cells.groupby("boro")["rate_diff"].rank(method="first", ascending=False)
    cells["persistent_score"] = cells["years_active"] * 2 + cells["years_ge3"] + np.log1p(cells["total_count"])
    cells["persistent_rank_boro"] = cells.groupby("boro")["persistent_score"].rank(method="first", ascending=False)
    cells["fading_rank_boro"] = cells.groupby("boro")["rate_diff"].rank(method="first", ascending=True)
    return cells


def select_views(cells):
    views = []
    for boro, sub in cells.groupby("boro"):
        recent = sub[(sub["recent_count"] >= 5)].sort_values("recent_count", ascending=False).head(25)
        emerging = sub[(sub["recent_count"] >= 6) & (sub["rate_diff"] > 0)].sort_values("rate_diff", ascending=False).head(25)
        persistent = sub[(sub["years_active"] >= 8)].sort_values("persistent_score", ascending=False).head(25)
        fading = sub[(sub["early_count"] >= 8) & (sub["rate_diff"] < 0)].sort_values("rate_diff", ascending=True).head(25)
        for label, frame in [
            ("Recent hotspots", recent),
            ("Emerging hotspots", emerging),
            ("Persistent hotspots", persistent),
            ("Fading hotspots", fading),
        ]:
            tmp = frame.copy()
            tmp["view"] = label
            views.append(tmp)
    return pd.concat(views, ignore_index=True)


def hover_text(df):
    return [
        (
            f"{r.boro} | Precinct {int(r.precinct)}"
            f"<br>250m hotspot cell"
            f"<br>Total incidents: {int(r.total_count):,}"
            f"<br>Recent 2023-2025: {int(r.recent_count):,} ({r.recent_rate:.1f}/yr)"
            f"<br>Early 2006-2011: {int(r.early_count):,} ({r.early_rate:.1f}/yr)"
            f"<br>Rate change: {r.rate_diff:+.1f}/yr"
            f"<br>Active years: {int(r.years_active)}"
            f"<br>Dominant location: {r.loc_class}"
            f"<br>Street/Housing: {r.street_share:.0f}% / {r.housing_share:.0f}%"
            f"<br>8p-3a share: {r.late_share:.0f}%"
            f"<br>lat {r.lat:.5f}, lon {r.lon:.5f}"
        )
        for r in df.itertuples(index=False)
    ]


def make_plot(df, selected):
    boroughs = list(BORO_COLORS)
    views = list(VIEW_COLORS)
    traces = []

    for boro, sub in df.groupby("BORO"):
        sample = sub.sample(min(2500, len(sub)), random_state=21)
        traces.append(
            go.Scattermapbox(
                lon=sample["lon"],
                lat=sample["lat"],
                mode="markers",
                marker=dict(size=3, color=BORO_COLORS[boro], opacity=0.08),
                name=f"{boro} incident sample",
                hoverinfo="skip",
                visible=True,
            )
        )

    trace_lookup = {}
    for boro in boroughs:
        for view in views:
            sub = selected[(selected["boro"] == boro) & (selected["view"] == view)]
            metric = {
                "Recent hotspots": "recent_count",
                "Emerging hotspots": "rate_diff",
                "Persistent hotspots": "years_active",
                "Fading hotspots": "rate_diff",
            }[view]
            size = np.clip(np.sqrt(np.maximum(sub[metric].abs(), 0.1)) * 11, 10, 44) if len(sub) else []
            traces.append(
                go.Scattermapbox(
                    lon=sub["lon"],
                    lat=sub["lat"],
                    mode="markers",
                    marker=dict(
                        size=size,
                        color=VIEW_COLORS[view],
                        opacity=0.78,
                    ),
                    name=f"{boro} - {view}",
                    text=hover_text(sub),
                    hoverinfo="text",
                    visible=view == "Recent hotspots",
                )
            )
            trace_lookup[(boro, view)] = len(traces) - 1

    def visibility(selected_boro, selected_view):
        vis = [False] * len(traces)
        for i, boro in enumerate(boroughs):
            vis[i] = selected_boro == "ALL" or selected_boro == boro
        for boro in boroughs:
            idx = trace_lookup[(boro, selected_view)]
            vis[idx] = selected_boro == "ALL" or selected_boro == boro
        return vis

    buttons = []
    for view in views:
        buttons.append(
            {
                "label": view,
                "method": "update",
                "args": [
                    {"visible": visibility("ALL", view)},
                    {"title": f"Actionable 250m Shooting Hotspots: {view}"},
                ],
            }
        )

    boro_buttons = [
        {
            "label": "All Boroughs",
            "method": "update",
            "args": [{"visible": visibility("ALL", "Recent hotspots")}],
        }
    ]
    for boro in boroughs:
        boro_buttons.append(
            {
                "label": boro.title(),
                "method": "update",
                "args": [{"visible": visibility(boro, "Recent hotspots")}],
            }
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Actionable 250m Shooting Hotspots: Recent hotspots",
        template="plotly_dark",
        width=1120,
        height=920,
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lon=df["lon"].mean(), lat=df["lat"].mean()),
            zoom=9.5,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.0,
                "y": 1.10,
                "showactive": True,
                "buttons": buttons,
            },
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.0,
                "y": 1.04,
                "showactive": True,
                "buttons": boro_buttons,
            },
        ],
        annotations=[
            dict(
                text=(
                    "Better than centroid: each marker is an actual 250m cell with count, trend, "
                    "persistence, precinct, location mix, and time profile. Hover markers for details."
                ),
                x=0,
                y=0.995,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                font=dict(size=13, color="#c9d1d9"),
            )
        ],
    )
    out = OUTPUT_DIR / "actionable_hotspot_explorer.html"
    fig.write_html(out, include_plotlyjs=True)
    return out


def write_summary(selected):
    top = selected[selected["view"].eq("Emerging hotspots")].sort_values("rate_diff", ascending=False).head(12)
    recent = selected[selected["view"].eq("Recent hotspots")].sort_values("recent_count", ascending=False).head(12)
    persistent = selected[selected["view"].eq("Persistent hotspots")].sort_values("persistent_score", ascending=False).head(12)

    lines = [
        "# Actionable Hotspot Method",
        "",
        "A centroid is useful for summarizing directional drift, but it is not a deployable location.",
        "This view uses 250m hotspot cells instead. Each cell has an incident count, recent-vs-early trend, active-year persistence, precinct, dominant location class, and late-night share.",
        "",
        "## Top Recent 250m Cells",
        "",
    ]
    for r in recent.itertuples(index=False):
        lines.append(
            f"- {r.boro} P{int(r.precinct)} at {r.lat:.5f}, {r.lon:.5f}: {int(r.recent_count)} incidents in 2023-2025, {r.recent_rate:.1f}/yr, {int(r.years_active)} active years."
        )
    lines.extend(["", "## Fastest Emerging 250m Cells", ""])
    for r in top.itertuples(index=False):
        lines.append(
            f"- {r.boro} P{int(r.precinct)} at {r.lat:.5f}, {r.lon:.5f}: rate change {r.rate_diff:+.1f}/yr, recent {int(r.recent_count)} vs early {int(r.early_count)}."
        )
    lines.extend(["", "## Most Persistent 250m Cells", ""])
    for r in persistent.itertuples(index=False):
        lines.append(
            f"- {r.boro} P{int(r.precinct)} at {r.lat:.5f}, {r.lon:.5f}: {int(r.total_count)} total incidents across {int(r.years_active)} active years."
        )
    (OUTPUT_DIR / "actionable_hotspot_summary.md").write_text("\n".join(lines))


def main():
    df = load_points()
    cells = build_cells(df)
    selected = select_views(cells)
    cells.to_csv(OUTPUT_DIR / "hotspot_250m_cells.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "hotspot_250m_selected_views.csv", index=False)
    out = make_plot(df, selected)
    write_summary(selected)
    print(f"Wrote {out}")
    print(f"Wrote {OUTPUT_DIR / 'hotspot_250m_cells.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'actionable_hotspot_summary.md'}")


if __name__ == "__main__":
    main()
