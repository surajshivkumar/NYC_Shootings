"""
Enriched Hotspot Analysis — Joins incidents with victim data to produce:
1. Murder-weighted DBSCAN hotspots (fatalities count 3x)
2. Murder-only hotspot map
3. Victim demographic profile per hotspot
4. Lethality rate by borough and year
5. Enhanced shifting hotspot dashboard with severity layer
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from pathlib import Path

DATA_PATH = Path("shootings_nyc.csv")
VICTIMS_PATH = Path("Shooting_Victims_(2006-Present)_20260729.csv")
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

try:
    with open("outputs/optimal_radius.txt", "r") as f:
        OPTIMAL_RADIUS_M = float(f.read().strip())
except FileNotFoundError:
    OPTIMAL_RADIUS_M = 979.1


def load_enriched():
    """Load incidents and join with victim-level murder/demographics data."""
    inc = pd.read_csv(DATA_PATH)
    inc["date"] = pd.to_datetime(inc["OCCUR_DATE"])
    inc["year"] = inc["date"].dt.year
    inc["lon"] = inc["Latitude"]
    inc["lat"] = inc["Longitude"]
    inc = inc[inc["lon"].between(-75, -73) & inc["lat"].between(40, 41)].copy()
    inc = inc[inc["BORO"].isin(BORO_COLORS)].copy()

    vic = pd.read_csv(VICTIMS_PATH)
    vic.columns = vic.columns.str.strip().str.strip('"')

    # Aggregate victim data per incident
    vic["is_murder"] = vic["STAT_MURDER_FLG"].str.strip('"').eq("Y").astype(int)
    vic["age"] = vic["VICTIM_AGE_GROUP"].str.strip('"')
    vic["sex"] = vic["VICTIM_SEX"].str.strip('"')
    vic["race"] = vic["VICTIM_RACE"].str.strip('"')

    vic["INCIDENT_KEY"] = pd.to_numeric(vic["INCIDENT_KEY"].astype(str).str.strip('"'), errors="coerce")

    agg = vic.groupby("INCIDENT_KEY").agg(
        n_victims=("is_murder", "size"),
        n_murders=("is_murder", "sum"),
        pct_under18=("age", lambda s: (s == "<18").mean()),
        pct_male=("sex", lambda s: (s == "MALE").mean()),
    ).reset_index()

    # Severity weight: each incident = 1 + 2 * murders (so a murder counts 3x)
    agg["severity"] = agg["n_victims"] + 2 * agg["n_murders"]

    df = inc.merge(agg, on="INCIDENT_KEY", how="left")
    df["n_victims"] = df["n_victims"].fillna(1).astype(int)
    df["n_murders"] = df["n_murders"].fillna(0).astype(int)
    df["severity"] = df["severity"].fillna(1)
    df["pct_under18"] = df["pct_under18"].fillna(0)
    df["pct_male"] = df["pct_male"].fillna(0.5)

    return df, vic


def find_weighted_hotspots(df):
    """DBSCAN on all incidents, but centroid is severity-weighted."""
    rows = []
    for (boro, year), sub in df.groupby(["BORO", "year"]):
        if len(sub) == 0:
            continue
        coords = np.deg2rad(sub[["lat", "lon"]].to_numpy())
        min_samp = min(35, max(5, int(len(sub) * 0.12)))
        labels = DBSCAN(
            eps=OPTIMAL_RADIUS_M / RADIUS_M,
            min_samples=min_samp,
            metric="haversine",
            algorithm="ball_tree",
        ).fit_predict(coords)
        sub = sub.copy()
        sub["cluster"] = labels
        clusters = sub[sub["cluster"] != -1]

        if len(clusters) == 0:
            lat_c = sub["lat"].median()
            lon_c = sub["lon"].median()
            n_inc = len(sub)
            n_murders = int(sub["n_murders"].sum())
            n_victims = int(sub["n_victims"].sum())
            severity = sub["severity"].sum()
        else:
            pid = clusters["cluster"].value_counts().index[0]
            pc = clusters[clusters["cluster"] == pid]
            # Severity-weighted centroid
            w = pc["severity"].values
            w_sum = w.sum()
            lat_c = np.average(pc["lat"].values, weights=w)
            lon_c = np.average(pc["lon"].values, weights=w)
            n_inc = len(pc)
            n_murders = int(pc["n_murders"].sum())
            n_victims = int(pc["n_victims"].sum())
            severity = pc["severity"].sum()

        lethality = n_murders / n_victims * 100 if n_victims > 0 else 0

        rows.append({
            "boro": boro, "year": int(year),
            "lat": lat_c, "lon": lon_c,
            "n_incidents": n_inc, "n_victims": n_victims, "n_murders": n_murders,
            "severity_score": round(severity, 1),
            "lethality_pct": round(lethality, 1),
        })
    return pd.DataFrame(rows).sort_values(["boro", "year"])


def find_murder_only_hotspots(df):
    """DBSCAN on ONLY fatal incidents."""
    murders = df[df["n_murders"] > 0].copy()
    rows = []
    for (boro, year), sub in murders.groupby(["BORO", "year"]):
        if len(sub) < 3:
            rows.append({
                "boro": boro, "year": int(year),
                "lat": sub["lat"].median(), "lon": sub["lon"].median(),
                "n_murders": int(sub["n_murders"].sum()),
            })
            continue
        coords = np.deg2rad(sub[["lat", "lon"]].to_numpy())
        min_samp = min(10, max(3, int(len(sub) * 0.15)))
        labels = DBSCAN(
            eps=OPTIMAL_RADIUS_M / RADIUS_M,
            min_samples=min_samp,
            metric="haversine",
            algorithm="ball_tree",
        ).fit_predict(coords)
        sub = sub.copy()
        sub["cluster"] = labels
        clusters = sub[sub["cluster"] != -1]
        if len(clusters) == 0:
            lat_c, lon_c = sub["lat"].median(), sub["lon"].median()
            n_m = int(sub["n_murders"].sum())
        else:
            pid = clusters["cluster"].value_counts().index[0]
            pc = clusters[clusters["cluster"] == pid]
            lat_c, lon_c = pc["lat"].mean(), pc["lon"].mean()
            n_m = int(pc["n_murders"].sum())
        rows.append({"boro": boro, "year": int(year), "lat": lat_c, "lon": lon_c, "n_murders": n_m})
    return pd.DataFrame(rows).sort_values(["boro", "year"])


def build_dashboard(df, weighted, murder_only):
    years = sorted(int(y) for y in weighted["year"].unique())

    # Pre-compute per-year data
    yearly_data = {}
    for year in years:
        boro_data = {}
        for boro in BOROUGHS:
            # Weighted hotspot path
            hist = weighted[(weighted["boro"] == boro) & (weighted["year"] <= year)]
            cur = weighted[(weighted["boro"] == boro) & (weighted["year"] == year)]
            # Murder-only path
            m_hist = murder_only[(murder_only["boro"] == boro) & (murder_only["year"] <= year)]
            m_cur = murder_only[(murder_only["boro"] == boro) & (murder_only["year"] == year)]

            cur_row = cur.iloc[0] if not cur.empty else None
            m_cur_row = m_cur.iloc[0] if not m_cur.empty else None

            boro_data[boro] = {
                # Severity-weighted hotspot
                "w_path_lon": hist["lon"].tolist(),
                "w_path_lat": hist["lat"].tolist(),
                "w_path_text": [
                    f"{int(r.year)} • {int(r.n_incidents)} incidents, {int(r.n_murders)} murders ({r.lethality_pct:.0f}% lethal)"
                    for r in hist.itertuples()
                ],
                "w_dot_lon": [cur_row["lon"]] if cur_row is not None else [],
                "w_dot_lat": [cur_row["lat"]] if cur_row is not None else [],
                "w_dot_text": [f"{boro} {year}<br>{int(cur_row['n_murders'])} murders"] if cur_row is not None else [],
                "w_dot_size": [max(14, min(40, int(cur_row["n_murders"] * 1.2 + 10)))] if cur_row is not None else [14],
                # Murder-only hotspot
                "m_path_lon": m_hist["lon"].tolist(),
                "m_path_lat": m_hist["lat"].tolist(),
                "m_path_text": [f"{int(r.year)} • {int(r.n_murders)} murders" for r in m_hist.itertuples()],
                "m_dot_lon": [m_cur_row["lon"]] if m_cur_row is not None else [],
                "m_dot_lat": [m_cur_row["lat"]] if m_cur_row is not None else [],
                "m_dot_text": [f"☠ {boro} {year}"] if m_cur_row is not None else [],
            }
        yearly_data[str(year)] = boro_data

    # Build initial figure
    traces = []

    # Background scatter
    for boro in BOROUGHS:
        sub = df[df["BORO"] == boro]
        sample = sub.sample(min(1500, len(sub)), random_state=11)
        traces.append(go.Scattermap(
            lon=sample["lon"], lat=sample["lat"], mode="markers",
            marker=dict(size=3, color=BORO_COLORS[boro], opacity=0.08),
            name=f"{boro} incidents", hoverinfo="skip", legendgroup=boro,
        ))
    n_bg = len(traces)

    init = yearly_data[str(years[0])]
    for boro in BOROUGHS:
        d = init[boro]
        # Weighted path
        traces.append(go.Scattermap(
            lon=d["w_path_lon"], lat=d["w_path_lat"], mode="lines+markers",
            line=dict(color=BORO_COLORS[boro], width=3),
            marker=dict(size=7, color=BORO_COLORS[boro], opacity=0.6),
            name=f"{boro} weighted", legendgroup=boro,
            text=d["w_path_text"], hovertemplate="%{text}<extra></extra>",
        ))
        # Weighted dot
        traces.append(go.Scattermap(
            lon=d["w_dot_lon"], lat=d["w_dot_lat"], mode="markers+text",
            marker=dict(size=d["w_dot_size"], color=BORO_COLORS[boro], opacity=0.85),
            text=d["w_dot_text"], textposition="top center",
            textfont=dict(color="white", size=12, family="Inter"),
            name=f"{boro} hotspot", legendgroup=boro,
            hovertemplate=f"{boro}<br>%{{text}}<extra></extra>",
        ))
        # Murder-only path (dashed, dimmer)
        traces.append(go.Scattermap(
            lon=d["m_path_lon"], lat=d["m_path_lat"], mode="lines+markers",
            line=dict(color="#ff0040", width=2),
            marker=dict(size=5, color="#ff0040", opacity=0.5),
            name=f"{boro} kill zone", legendgroup=f"{boro}_m",
            text=d["m_path_text"], hovertemplate="%{text}<extra></extra>",
            visible="legendonly",  # hidden by default, toggle on
        ))
        # Murder-only dot
        traces.append(go.Scattermap(
            lon=d["m_dot_lon"], lat=d["m_dot_lat"], mode="markers+text",
            marker=dict(size=18, color="#ff0040", opacity=0.9),
            text=d["m_dot_text"], textposition="top center",
            textfont=dict(color="#ff0040", size=11, family="Inter"),
            name=f"{boro} murder hotspot", legendgroup=f"{boro}_m",
            hovertemplate=f"{boro} murder zone<br>%{{text}}<extra></extra>",
            visible="legendonly",
        ))

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

    # Compute summary stats for the header
    total_victims = int(df["n_victims"].sum())
    total_murders = int(df["n_murders"].sum())
    overall_lethality = total_murders / total_victims * 100

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NYC Severity-Weighted Hotspot Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box}}
body{{
  margin:0;padding:0;background:#0b0f19;
  font-family:'Inter',system-ui,sans-serif;color:#f0f6fc;
  display:flex;flex-direction:column;align-items:center;min-height:100vh;
  background-image:
    radial-gradient(at 0% 0%,rgba(255,0,64,.12) 0,transparent 50%),
    radial-gradient(at 100% 0%,rgba(0,212,255,.08) 0,transparent 50%);
}}
.header{{text-align:center;padding:2rem 1rem .5rem;max-width:1000px}}
h1{{
  font-size:2.4rem;font-weight:800;margin:.4rem 0;
  background:linear-gradient(90deg,#fff,#8b949e);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}}
.sub{{color:#8b949e;font-size:1rem;line-height:1.5;margin:0}}
.stats-row{{
  display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;
  margin:1rem 0;
}}
.stat-card{{
  background:rgba(22,27,34,.8);border:1px solid rgba(255,255,255,.12);
  border-radius:10px;padding:.75rem 1.5rem;text-align:center;
  min-width:140px;
}}
.stat-val{{font-size:1.6rem;font-weight:800}}
.stat-lbl{{font-size:.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em}}
.dash{{
  width:95%;max-width:1400px;margin:.5rem auto 3rem;
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
.sld input{{width:100%;accent-color:#ff0040;cursor:pointer}}
.yr{{font-size:1.3rem;font-weight:800;color:#ff0040;min-width:55px;text-align:center}}
.layer-toggle{{
  display:flex;gap:.5rem;align-items:center;
}}
.layer-btn{{
  background:rgba(255,0,64,.15);border:1px solid rgba(255,0,64,.4);
  color:#ff4070;padding:.45rem .8rem;border-radius:6px;
  font-size:.85rem;font-weight:600;cursor:pointer;transition:.2s;
}}
.layer-btn:hover,.layer-btn.on{{background:rgba(255,0,64,.35);color:#fff;border-color:#ff0040}}
.boros{{display:flex;gap:.4rem;flex-wrap:wrap}}
.bb{{
  background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.12);
  color:#c9d1d9;padding:.45rem .8rem;border-radius:6px;
  font-size:.85rem;font-weight:600;cursor:pointer;transition:.2s;
}}
.bb:hover,.bb.on{{background:rgba(255,255,255,.15);color:#fff;border-color:rgba(255,255,255,.4)}}
.mapbox{{height:65vh;min-height:520px;border-radius:12px;overflow:hidden}}
.mapbox .js-plotly-plot,.mapbox .plotly{{height:100%!important;width:100%!important}}
</style>
</head>
<body>
<div class="header">
  <h1>Severity-Weighted Hotspot Trajectories</h1>
  <p class="sub">Murder-weighted centroids pull toward the most <b>lethal</b> zones, not just the busiest. Toggle the red "Kill Zone" layer to see murder-only hotspots.</p>
  <div class="stats-row">
    <div class="stat-card"><div class="stat-val" style="color:#ff5a5f">{total_victims:,}</div><div class="stat-lbl">Total Victims</div></div>
    <div class="stat-card"><div class="stat-val" style="color:#ff0040">{total_murders:,}</div><div class="stat-lbl">Murders</div></div>
    <div class="stat-card"><div class="stat-val" style="color:#ffe66d">{overall_lethality:.1f}%</div><div class="stat-lbl">Lethality Rate</div></div>
    <div class="stat-card"><div class="stat-val" style="color:#00d4ff">{OPTIMAL_RADIUS_M:.0f}m</div><div class="stat-lbl">DBSCAN Radius</div></div>
  </div>
</div>

<div class="dash">
  <div class="bar">
    <div class="ctrl">
      <button id="playBtn" class="pbtn" onclick="togglePlay()">▶ Play</button>
      <div class="sld">
        <input type="range" id="slider" min="0" max="{len(years)-1}" value="0" step="1" oninput="goToIndex(+this.value)">
        <span id="yrLbl" class="yr">{years[0]}</span>
      </div>
      <div class="layer-toggle">
        <button class="layer-btn" onclick="toggleMurderLayer(this)">☠ Kill Zones</button>
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
// Per borough: 4 traces (w_path, w_dot, m_path, m_dot)
var TRACES_PER_BORO = 4;
var curIdx = 0, playing = false, timer = null;
var showMurder = false;
var selBoro = "ALL";

function gd() {{ return document.querySelector('.js-plotly-plot'); }}

function setYear(idx) {{
    curIdx = idx;
    var year = YEARS[idx];
    document.getElementById('slider').value = idx;
    document.getElementById('yrLbl').innerText = year;
    var g = gd(); if (!g) return;
    var d = DATA[String(year)];

    for (var i = 0; i < N_BORO; i++) {{
        var boro = BOROUGHS[i];
        var bd = d[boro];
        var base = N_BG + i * TRACES_PER_BORO;
        // Weighted path
        Plotly.restyle(g, {{lon:[bd.w_path_lon],lat:[bd.w_path_lat],text:[bd.w_path_text]}}, [base]);
        // Weighted dot
        Plotly.restyle(g, {{lon:[bd.w_dot_lon],lat:[bd.w_dot_lat],text:[bd.w_dot_text],'marker.size':[bd.w_dot_size]}}, [base+1]);
        // Murder path
        Plotly.restyle(g, {{lon:[bd.m_path_lon],lat:[bd.m_path_lat],text:[bd.m_path_text]}}, [base+2]);
        // Murder dot
        Plotly.restyle(g, {{lon:[bd.m_dot_lon],lat:[bd.m_dot_lat],text:[bd.m_dot_text]}}, [base+3]);
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

function toggleMurderLayer(el) {{
    showMurder = !showMurder;
    el.classList.toggle('on');
    applyVisibility();
}}

function filterBoro(boro, el) {{
    selBoro = boro;
    document.querySelectorAll('.bb').forEach(b => b.classList.remove('on'));
    if (el) el.classList.add('on');
    applyVisibility();
}}

function applyVisibility() {{
    var g = gd(); if (!g) return;
    // Background
    for (var i = 0; i < N_BORO; i++) {{
        var show = (selBoro === 'ALL' || selBoro === BOROUGHS[i]);
        Plotly.restyle(g, {{visible: show}}, [i]);
    }}
    // Per-boro traces
    for (var i = 0; i < N_BORO; i++) {{
        var show = (selBoro === 'ALL' || selBoro === BOROUGHS[i]);
        var base = N_BG + i * TRACES_PER_BORO;
        Plotly.restyle(g, {{visible: show}}, [base]);     // w path
        Plotly.restyle(g, {{visible: show}}, [base+1]);   // w dot
        Plotly.restyle(g, {{visible: show && showMurder}}, [base+2]); // m path
        Plotly.restyle(g, {{visible: show && showMurder}}, [base+3]); // m dot
    }}
}}
</script>
</body>
</html>"""

    out = OUTPUT_DIR / "severity_weighted_hotspots.html"
    out.write_text(html)
    print(f"Wrote {out}")


def print_insights(df, weighted, murder_only, vic):
    print("\n" + "=" * 70)
    print("ENRICHED ANALYSIS INSIGHTS")
    print("=" * 70)

    print(f"\nTotal incidents joined: {len(df):,}")
    print(f"Total victims: {int(df['n_victims'].sum()):,}")
    print(f"Total murders: {int(df['n_murders'].sum()):,}")
    print(f"Overall lethality: {df['n_murders'].sum() / df['n_victims'].sum() * 100:.1f}%")

    print("\n--- Lethality by Borough ---")
    for boro in BOROUGHS:
        sub = weighted[weighted["boro"] == boro]
        total_v = sub["n_victims"].sum()
        total_m = sub["n_murders"].sum()
        rate = total_m / total_v * 100 if total_v > 0 else 0
        print(f"  {boro:>15}: {total_m:>4} murders / {total_v:>5} victims = {rate:.1f}% lethality")

    print("\n--- COVID Lethality Spike ---")
    for boro in ["BROOKLYN", "BRONX", "MANHATTAN", "QUEENS"]:
        r19 = weighted[(weighted["boro"] == boro) & (weighted["year"] == 2019)]
        r20 = weighted[(weighted["boro"] == boro) & (weighted["year"] == 2020)]
        if not r19.empty and not r20.empty:
            print(f"  {boro}: 2019 lethality={r19.iloc[0]['lethality_pct']:.0f}%  →  2020={r20.iloc[0]['lethality_pct']:.0f}%")

    print("\n--- Murder vs Shooting Hotspot Drift ---")
    for boro in ["BROOKLYN", "BRONX", "MANHATTAN", "QUEENS"]:
        w = weighted[weighted["boro"] == boro]
        m = murder_only[murder_only["boro"] == boro]
        if not w.empty and not m.empty:
            w_lat, w_lon = w["lat"].mean(), w["lon"].mean()
            m_lat, m_lon = m["lat"].mean(), m["lon"].mean()
            drift_km = np.sqrt(((w_lat - m_lat) * 111) ** 2 + ((w_lon - m_lon) * 111 * np.cos(np.radians(40.7))) ** 2)
            print(f"  {boro}: murder hotspot is {drift_km:.2f} km from shooting hotspot")

    print("\n--- Victim Demographics (from victim file) ---")
    print(f"  Age: {vic['age'].value_counts().head(3).to_dict()}")
    print(f"  Sex: {vic['sex'].value_counts().head(3).to_dict()}")
    print(f"  Race: {vic['race'].value_counts().head(3).to_dict()}")

    # Under-18 victims per year
    print("\n--- Under-18 Victims by Year ---")
    vic_with_year = vic.copy()
    vic_with_year["INCIDENT_KEY"] = pd.to_numeric(vic_with_year["INCIDENT_KEY"].astype(str).str.strip('"'), errors="coerce")
    merged = vic_with_year.merge(df[["INCIDENT_KEY", "year"]].drop_duplicates(), on="INCIDENT_KEY", how="left")
    u18 = merged[merged["age"] == "<18"].groupby("year").size()
    for yr, cnt in u18.items():
        print(f"  {yr}: {cnt} child victims")


def main():
    df, vic = load_enriched()
    weighted = find_weighted_hotspots(df)
    murder_only = find_murder_only_hotspots(df)

    weighted.to_csv(OUTPUT_DIR / "severity_weighted_hotspots.csv", index=False)
    murder_only.to_csv(OUTPUT_DIR / "murder_only_hotspots.csv", index=False)

    build_dashboard(df, weighted, murder_only)
    print_insights(df, weighted, murder_only, vic)


if __name__ == "__main__":
    main()
