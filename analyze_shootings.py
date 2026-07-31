from pathlib import Path
from html import escape

import pandas as pd


DATA_PATH = Path("shootings_nyc.csv")
OUTPUT_DIR = Path("outputs")
REPORT_PATH = OUTPUT_DIR / "nypd_shootings_insights.md"


def pct(value: float) -> str:
    return f"{value:.1f}%"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["OCCUR_DATE"], errors="coerce")

    raw_time = df["OCCUR_TIME"].astype(str)
    hhmmss_hour = pd.to_numeric(raw_time.str.extract(r"^(\d{1,2}):")[0], errors="coerce")
    datetime_hour = pd.to_datetime(raw_time, errors="coerce", format="mixed").dt.hour
    df["hour"] = hhmmss_hour.fillna(datetime_hour).astype("Int64")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["period"] = pd.cut(
        df["hour"].astype(int),
        bins=[-1, 5, 11, 17, 23],
        labels=["Late night (12-5a)", "Morning (6-11a)", "Afternoon (12-5p)", "Evening (6-11p)"],
    )
    df["loc_class"] = df["LOC_CLASSFCTN_DESC"].replace({"(null)": pd.NA}).fillna("Unknown")
    df["location_desc"] = df["LOCATION_DESC"].replace({"(null)": pd.NA}).fillna("Unspecified")
    return df


def svg_bar(series, title, path, color="#2f6f7e", horizontal=False):
    series = series.dropna()
    width, height = 920, 500
    margin = {"top": 58, "right": 28, "bottom": 82, "left": 92}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    max_v = max(float(series.max()), 1.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700">{escape(title)}</text>',
        f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" stroke="#333"/>',
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" stroke="#333"/>',
    ]
    if horizontal:
        step = plot_h / len(series)
        for i, (label, value) in enumerate(series.items()):
            bar_w = (float(value) / max_v) * (plot_w - 150)
            y = margin["top"] + i * step + step * 0.16
            parts.append(f'<text x="22" y="{y + step * 0.45:.1f}" font-family="Arial" font-size="13">{escape(str(label))}</text>')
            parts.append(f'<rect x="{margin["left"]}" y="{y:.1f}" width="{bar_w:.1f}" height="{step * 0.62:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{margin["left"] + bar_w + 8:.1f}" y="{y + step * 0.43:.1f}" font-family="Arial" font-size="12">{int(value):,}</text>')
    else:
        step = plot_w / len(series)
        for i, (label, value) in enumerate(series.items()):
            bar_h = (float(value) / max_v) * plot_h
            x = margin["left"] + i * step + step * 0.14
            y = margin["top"] + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{step * 0.72:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{x + step * 0.36:.1f}" y="{height - 45}" font-family="Arial" font-size="12" text-anchor="middle">{escape(str(label))}</text>')
            if len(series) <= 12:
                parts.append(f'<text x="{x + step * 0.36:.1f}" y="{y - 6:.1f}" font-family="Arial" font-size="12" text-anchor="middle">{int(value):,}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def svg_line(series, title, path, color="#2f6f7e"):
    width, height = 920, 500
    margin = {"top": 58, "right": 36, "bottom": 62, "left": 76}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    max_v = max(float(series.max()), 1.0)
    min_v = 0
    xs = [margin["left"] + i * plot_w / (len(series) - 1) for i in range(len(series))]
    ys = [margin["top"] + plot_h - ((float(v) - min_v) / (max_v - min_v)) * plot_h for v in series]
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700">{escape(title)}</text>',
        f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" stroke="#333"/>',
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" stroke="#333"/>',
        f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4"/>',
    ]
    for x, y, label, value in zip(xs, ys, series.index, series):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
        if int(label) % 2 == 0:
            parts.append(f'<text x="{x:.1f}" y="{height - 30}" font-family="Arial" font-size="12" text-anchor="middle">{label}</text>')
    parts.append(f'<text x="18" y="{margin["top"] + 8}" font-family="Arial" font-size="12">{int(max_v):,}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_data()
    full_years = df[df["year"] < df["year"].max()]

    yearly = df.groupby("year").size()
    full_yearly = full_years.groupby("year").size()
    monthly = df.groupby("month").size().reindex(range(1, 13))
    monthly.index = pd.to_datetime(monthly.index, format="%m").month_name().str[:3]
    hourly = df.groupby("hour").size().reindex(range(24), fill_value=0)
    boro = df["BORO"].value_counts()
    boro_pct = boro / len(df) * 100
    precinct = df["PRECINCT"].value_counts().head(15)
    loc_class = df["loc_class"].value_counts()
    period = df["period"].value_counts().reindex(
        ["Late night (12-5a)", "Morning (6-11a)", "Afternoon (12-5p)", "Evening (6-11p)"]
    )

    svg_line(yearly, "NYC shooting incidents by year", OUTPUT_DIR / "01_yearly_trend.svg")
    svg_bar(boro, "Shooting incidents by borough", OUTPUT_DIR / "02_borough_totals.svg")
    svg_bar(hourly, "Shooting incidents by hour of day", OUTPUT_DIR / "03_hourly_pattern.svg", color="#6b5b95")
    svg_bar(monthly, "Shooting incidents by month", OUTPUT_DIR / "04_monthly_pattern.svg", color="#3b8b5f")
    svg_bar(precinct.sort_values(), "Top 15 precincts by incident count", OUTPUT_DIR / "05_top_precincts.svg", color="#b4573f", horizontal=True)
    svg_bar(loc_class.head(8), "Shooting incidents by location classification", OUTPUT_DIR / "06_location_class.svg", color="#466a9f")

    max_year = full_yearly.idxmax()
    pre_2020_low = full_yearly.loc[2019]
    spike_avg = full_yearly.loc[[2020, 2021]].mean()
    latest_full = full_yearly.index.max()
    latest_full_count = full_yearly.loc[latest_full]
    decline_from_2021 = (full_yearly.loc[2021] - latest_full_count) / full_yearly.loc[2021] * 100
    outside_share = df["LOC_OF_OCCUR_DESC"].eq("OUTSIDE").mean() * 100
    street_housing_share = loc_class.loc[["STREET", "HOUSING"]].sum() / len(df) * 100
    top_5_precinct_share = df["PRECINCT"].value_counts().head(5).sum() / len(df) * 100
    evening_share = period.loc["Evening (6-11p)"] / len(df) * 100
    late_evening_share = df["hour"].between(20, 23).mean() * 100
    summer_share = df["month"].isin([6, 7, 8, 9]).mean() * 100

    report = f"""# NYPD Shooting Incident Insights

Source: `{DATA_PATH}` with the NYPD shootings data dictionary. This incident table covers {df['date'].min().date()} through {df['date'].max().date()} and contains {len(df):,} unique incident records. The dictionary notes that victim and offender demographics are in separate related tables, so this analysis focuses on incident time and place.

## 1. Shootings fell for years, surged in 2020-2021, then declined again

Full-year incidents peaked at {full_yearly.loc[max_year]:,} in {max_year} and reached a pre-pandemic low of {pre_2020_low:,} in 2019. Incidents then roughly doubled to an average of {spike_avg:,.0f} in 2020-2021. By {latest_full}, incidents were down to {latest_full_count:,}, a {pct(decline_from_2021)} decrease from 2021. The 2026 count is year-to-date through {df['date'].max().date()}, so it should not be compared to full years.

Chart: `outputs/01_yearly_trend.svg`

## 2. Brooklyn and the Bronx dominate the incident count

Brooklyn accounts for {boro['BROOKLYN']:,} incidents ({pct(boro_pct['BROOKLYN'])}) and the Bronx for {boro['BRONX']:,} ({pct(boro_pct['BRONX'])}). Together, they represent {pct((boro['BROOKLYN'] + boro['BRONX']) / len(df) * 100)} of all recorded incidents. Queens and Manhattan are materially lower, while Staten Island is a small share.

Chart: `outputs/02_borough_totals.svg`

## 3. The risk window is heavily evening and late night

Evening hours from 6 p.m. to 11 p.m. account for {period.loc['Evening (6-11p)']:,} incidents ({pct(evening_share)}). The single busiest hours are 11 p.m. ({hourly.loc[23]:,}), 10 p.m. ({hourly.loc[22]:,}), and midnight ({hourly.loc[0]:,}). Hours 8 p.m. through 11 p.m. alone make up {pct(late_evening_share)} of the dataset.

Chart: `outputs/03_hourly_pattern.svg`

## 4. Shootings are seasonal, with a clear warm-weather lift

June through September account for {pct(summer_share)} of all incidents, higher than the one-third share expected if incidents were evenly distributed across months. July is the highest month at {monthly.loc['Jul']:,} incidents, followed by August at {monthly.loc['Aug']:,}.

Chart: `outputs/04_monthly_pattern.svg`

## 5. A small set of precincts carries a large share

The top five precincts by incident count are {', '.join(map(str, precinct.head(5).index.tolist()))}; together they account for {pct(top_5_precinct_share)} of all incidents. Precinct 75 has the highest count at {precinct.iloc[0]:,}, followed by precinct 73 at {precinct.iloc[1]:,}.

Chart: `outputs/05_top_precincts.svg`

## 6. Incidents are overwhelmingly outdoor and street/housing-related

Outdoor incidents total {df['LOC_OF_OCCUR_DESC'].value_counts().loc['OUTSIDE']:,}, or {pct(outside_share)} of all records. The `STREET` and `HOUSING` location classes together account for {pct(street_housing_share)} of incidents, making them the dominant environmental context in this table.

Chart: `outputs/06_location_class.svg`

## Notes

- `LOCATION_DESC` is missing or unspecified for many street incidents, so it is less reliable as a primary categorization field than `LOC_CLASSFCTN_DESC`.
- Borough values `PBXN` and `PBXS` appear as small Patrol Borough categories rather than standard borough names; they were left unchanged in the totals.
- Charts were generated by `analyze_shootings.py`.
"""
    REPORT_PATH.write_text(report)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote charts to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
