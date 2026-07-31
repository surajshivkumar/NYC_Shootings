"""
Borough Annual Centroid Drift — Interactive Dashboard

Uses Plotly.restyle() via custom JS (no Plotly frames) to guarantee
Play, Slider, and Borough filter buttons work flawlessly.
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

DATA_PATH = Path("shootings_nyc.csv")
OUTPUT_DIR = Path("outputs/spatial")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RADIUS_M = 6_371_008.8
BORO_COLORS = {
    "BRONX": "#00d4ff",
    "BROOKLYN": "#ff5a5f",
    "MANHATTAN": "#ffe66d",
    "QUEENS": "#7bed9f",
    "STATEN ISLAND": "#c56cf0",
}
BOROUGHS = list(BORO_COLORS.keys())


def load_points():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["OCCUR_DATE"])
    df["year"] = df["date"].dt.year
    df["lon"] = df["Latitude"]
    df["lat"] = df["Longitude"]
    df = df[df["lon"].between(-75, -73) & df["lat"].between(40, 41)].copy()
    df = df[df["BORO"].isin(BORO_COLORS)].copy()
    lat_ref = np.deg2rad(df["lat"].mean())
    lon_ref = np.deg2rad(df["lon"].mean())
    df.attrs["lat_ref"] = lat_ref
    df.attrs["lon_ref"] = lon_ref
    df["x_m"] = (np.deg2rad(df["lon"]) - lon_ref) * RADIUS_M * np.cos(lat_ref)
    df["y_m"] = (np.deg2rad(df["lat"]) - lat_ref) * RADIUS_M
    return df


def xy_to_lonlat(df, x, y):
    lat_ref = df.attrs["lat_ref"]
    lon_ref = df.attrs["lon_ref"]
    lon = np.rad2deg(x / (RADIUS_M * np.cos(lat_ref)) + lon_ref)
    lat = np.rad2deg(y / RADIUS_M + lat_ref)
    return lon, lat


def borough_annual_centers(df):
    rows = []
    for (boro, year), sub in df.groupby(["BORO", "year"]):
        lon, lat = xy_to_lonlat(df, sub["x_m"].mean(), sub["y_m"].mean())
        rows.append({"boro": boro, "year": int(year), "lat": lat, "lon": lon, "n": len(sub)})
    return pd.DataFrame(rows).sort_values(["boro", "year"])


def build_dashboard(df, centers):
    years = sorted(int(y) for y in centers["year"].unique())

    # Pre-compute per-year data
    yearly_data = {}
    for year in years:
        boro_data = {}
        for boro in BOROUGHS:
            hist = centers[(centers["boro"] == boro) & (centers["year"] <= year)]
            current = centers[(centers["boro"] == boro) & (centers["year"] == year)]
            boro_data[boro] = {
                "path_lon": hist["lon"].tolist(),
                "path_lat": hist["lat"].tolist(),
                "path_text": [f"{int(r.year)} • {int(r.n):,} incidents" for r in hist.itertuples()],
                "dot_lon": current["lon"].tolist(),
                "dot_lat": current["lat"].tolist(),
                "dot_text": [f"{boro} {year}"] if not current.empty else [],
            }
        yearly_data[str(year)] = boro_data

    # Build initial figure (NO frames)
    traces = []
    for boro in BOROUGHS:
        sub = df[df["BORO"] == boro]
        sample = sub.sample(min(1800, len(sub)), random_state=11) if len(sub) > 0 else sub
        traces.append(
            go.Scattermap(
                lon=sample["lon"], lat=sample["lat"],
                mode="markers",
                marker=dict(size=3, color=BORO_COLORS[boro], opacity=0.10),
                name=f"{boro} incidents", hoverinfo="skip", legendgroup=boro,
            )
        )
    n_bg = len(traces)
    init = yearly_data[str(years[0])]

    for boro in BOROUGHS:
        d = init[boro]
        traces.append(
            go.Scattermap(
                lon=d["path_lon"], lat=d["path_lat"],
                mode="lines+markers",
                line=dict(color=BORO_COLORS[boro], width=3),
                marker=dict(size=8, color=BORO_COLORS[boro], opacity=0.7),
                name=f"{boro} path", legendgroup=boro,
                text=d["path_text"], hovertemplate="%{text}<extra></extra>",
            )
        )
        traces.append(
            go.Scattermap(
                lon=d["dot_lon"], lat=d["dot_lat"],
                mode="markers+text",
                marker=dict(size=24, color=BORO_COLORS[boro], opacity=0.9),
                text=d["dot_text"], textposition="top center",
                textfont=dict(color="white", size=13, family="Inter"),
                name=f"{boro} center", legendgroup=boro,
                hovertemplate=f"{boro}<br>%{{text}}<extra></extra>",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        map=dict(style="carto-darkmatter", center=dict(lon=-73.935, lat=40.730), zoom=9.8),
        showlegend=True,
        legend=dict(
            itemsizing="constant", yanchor="top", y=0.95, xanchor="left", x=0.02,
            bgcolor="rgba(22,27,34,0.85)", bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1, font=dict(color="#f0f6fc", size=11),
        ),
    )

    plot_html = fig.to_html(full_html=False, include_plotlyjs=True, default_height="100%", default_width="100%")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Borough Centroid Drift</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box}}
body{{
  margin:0;padding:0;background:#0b0f19;
  font-family:'Inter',system-ui,sans-serif;color:#f0f6fc;
  display:flex;flex-direction:column;align-items:center;min-height:100vh;
  background-image:
    radial-gradient(at 0% 0%,rgba(88,166,255,.12) 0,transparent 50%),
    radial-gradient(at 100% 0%,rgba(125,237,159,.08) 0,transparent 50%);
}}
.header{{text-align:center;padding:2rem 1rem 1rem;max-width:900px}}
h1{{
  font-size:2.4rem;font-weight:800;margin:.4rem 0;
  background:linear-gradient(90deg,#fff,#8b949e);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}}
.sub{{color:#8b949e;font-size:1.05rem;line-height:1.5;margin:0}}
.dash{{
  width:95%;max-width:1400px;margin:1rem auto 3rem;
  background:rgba(22,27,34,.75);
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  border:1px solid rgba(255,255,255,.15);border-radius:16px;
  padding:1.25rem;box-shadow:0 20px 50px rgba(0,0,0,.5);
  display:flex;flex-direction:column;gap:1rem;
}}
.bar{{
  display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;
  background:rgba(13,17,23,.8);border:1px solid rgba(255,255,255,.12);
  border-radius:12px;padding:.85rem 1.25rem;gap:1rem;
}}
.ctrl{{display:flex;align-items:center;gap:1rem}}
.pbtn{{
  background:#238636;color:#fff;border:none;
  padding:.6rem 1.4rem;font-weight:700;font-size:.95rem;
  border-radius:8px;cursor:pointer;transition:.2s;
  box-shadow:0 4px 12px rgba(35,134,54,.3);
}}
.pbtn:hover{{background:#2ea043;transform:translateY(-1px)}}
.pbtn.on{{background:#da3633;box-shadow:0 4px 12px rgba(218,54,51,.3)}}
.sld{{display:flex;align-items:center;gap:.75rem;flex-grow:1;max-width:420px}}
.sld input{{width:100%;accent-color:#58a6ff;cursor:pointer}}
.yr{{font-size:1.3rem;font-weight:800;color:#58a6ff;min-width:55px;text-align:center}}
.boros{{display:flex;gap:.4rem;flex-wrap:wrap}}
.bb{{
  background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.12);
  color:#c9d1d9;padding:.45rem .8rem;border-radius:6px;
  font-size:.85rem;font-weight:600;cursor:pointer;transition:.2s;
}}
.bb:hover,.bb.on{{background:rgba(255,255,255,.15);color:#fff;border-color:rgba(255,255,255,.4)}}
.mapbox{{height:68vh;min-height:550px;border-radius:12px;overflow:hidden}}
.mapbox .js-plotly-plot,.mapbox .plotly{{height:100%!important;width:100%!important}}
</style>
</head>
<body>
<div class="header">
  <h1>Borough-Wise Shooting Centroid Drift</h1>
  <p class="sub">The annual geographic center of gravity for shootings across all 5 NYC boroughs, animated year by year.</p>
</div>

<div class="dash">
  <div class="bar">
    <div class="ctrl">
      <button id="playBtn" class="pbtn" onclick="togglePlay()">▶ Play</button>
      <div class="sld">
        <input type="range" id="slider" min="0" max="{len(years)-1}" value="0" step="1"
               oninput="goToIndex(+this.value)">
        <span id="yrLbl" class="yr">{years[0]}</span>
      </div>
    </div>
    <div class="boros">
      <button class="bb on" onclick="filterBoro('ALL',this)">All</button>
      <button class="bb" style="border-left:3px solid #00d4ff" onclick="filterBoro('BRONX',this)">Bronx</button>
      <button class="bb" style="border-left:3px solid #ff5a5f" onclick="filterBoro('BROOKLYN',this)">Brooklyn</button>
      <button class="bb" style="border-left:3px solid #ffe66d" onclick="filterBoro('MANHATTAN',this)">Manhattan</button>
      <button class="bb" style="border-left:3px solid #7bed9f" onclick="filterBoro('QUEENS',this)">Queens</button>
      <button class="bb" style="border-left:3px solid #c56cf0" onclick="filterBoro('STATEN ISLAND',this)">S.I.</button>
    </div>
  </div>
  <div class="mapbox">{plot_html}</div>
</div>

<script>
var YEARS    = {json.dumps(years)};
var BOROUGHS = {json.dumps(BOROUGHS)};
var DATA     = {json.dumps(yearly_data)};
var N_BG     = {n_bg};
var N_BORO   = BOROUGHS.length;
var curIdx   = 0;
var playing  = false;
var timer    = null;

function gd() {{ return document.querySelector('.js-plotly-plot'); }}

function setYear(idx) {{
    curIdx = idx;
    var year = YEARS[idx];
    document.getElementById('slider').value = idx;
    document.getElementById('yrLbl').innerText = year;
    var g = gd();
    if (!g) return;
    var d = DATA[String(year)];
    for (var i = 0; i < N_BORO; i++) {{
        var boro = BOROUGHS[i];
        var bd = d[boro];
        Plotly.restyle(g, {{lon:[bd.path_lon],lat:[bd.path_lat],text:[bd.path_text]}}, [N_BG+i*2]);
        Plotly.restyle(g, {{lon:[bd.dot_lon],lat:[bd.dot_lat],text:[bd.dot_text]}}, [N_BG+i*2+1]);
    }}
}}
function goToIndex(idx) {{ setYear(idx); }}

function togglePlay() {{
    var btn = document.getElementById('playBtn');
    if (playing) {{
        playing = false; clearInterval(timer);
        btn.innerText = '▶ Play'; btn.classList.remove('on');
    }} else {{
        playing = true;
        btn.innerText = '❚❚ Pause'; btn.classList.add('on');
        timer = setInterval(function() {{
            curIdx = (curIdx + 1) % YEARS.length;
            setYear(curIdx);
        }}, 700);
    }}
}}

function filterBoro(boro, el) {{
    document.querySelectorAll('.bb').forEach(b => b.classList.remove('on'));
    if (el) el.classList.add('on');
    var g = gd(); if (!g) return;
    var vis = [];
    for (var i = 0; i < N_BORO; i++) vis.push((boro==='ALL'||boro===BOROUGHS[i])?true:'legendonly');
    for (var i = 0; i < N_BORO; i++) {{
        var show = (boro==='ALL'||boro===BOROUGHS[i])?true:'legendonly';
        vis.push(show); vis.push(show);
    }}
    for (var i = 0; i < vis.length; i++) Plotly.restyle(g, {{visible:vis[i]}}, [i]);
}}
</script>
</body>
</html>"""

    out = OUTPUT_DIR / "borough_center_click_through.html"
    out.write_text(html)
    print(f"Wrote {out}")
    return out


def main():
    df = load_points()
    centers = borough_annual_centers(df)
    centers.to_csv(OUTPUT_DIR / "borough_annual_centroids.csv", index=False)
    build_dashboard(df, centers)


if __name__ == "__main__":
    main()
