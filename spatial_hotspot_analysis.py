import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN


OUTPUT_DIR = Path("outputs/spatial")
DATA_PATH = Path("shootings_nyc.csv")
RADIUS_M = 6_371_008.8

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
cache_dir = OUTPUT_DIR / ".cache"
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm


def load_points() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["OCCUR_DATE"])
    df["year"] = df["date"].dt.year

    # The source file's coordinate values are swapped: Latitude has -73.x and Longitude has 40.x.
    df["lon"] = df["Latitude"]
    df["lat"] = df["Longitude"]
    df = df[df["lon"].between(-75, -73) & df["lat"].between(40, 41)].copy()

    raw_time = df["OCCUR_TIME"].astype(str)
    hhmmss_hour = pd.to_numeric(raw_time.str.extract(r"^(\d{1,2}):")[0], errors="coerce")
    datetime_hour = pd.to_datetime(raw_time, errors="coerce", format="mixed").dt.hour
    df["hour"] = hhmmss_hour.fillna(datetime_hour).astype(int)
    df["loc_class"] = df["LOC_CLASSFCTN_DESC"].replace({"(null)": pd.NA}).fillna("Unknown")

    lat0 = np.deg2rad(df["lat"].mean())
    lon_ref = np.deg2rad(df["lon"].mean())
    lat_ref = np.deg2rad(df["lat"].mean())
    df["x_m"] = (np.deg2rad(df["lon"]) - lon_ref) * RADIUS_M * np.cos(lat0)
    df["y_m"] = (np.deg2rad(df["lat"]) - lat_ref) * RADIUS_M
    return df


def add_dark_axes(ax, title):
    ax.set_facecolor("#0d1117")
    ax.figure.set_facecolor("#0d1117")
    ax.set_title(title, loc="left", color="white", fontsize=18, fontweight="bold", pad=16)
    ax.tick_params(colors="#c9d1d9", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(color="#30363d", alpha=0.35, linewidth=0.6)
    ax.set_xlabel("Longitude", color="#c9d1d9")
    ax.set_ylabel("Latitude", color="#c9d1d9")


def set_nyc_bounds(ax, df):
    ax.set_xlim(df["lon"].quantile(0.002) - 0.01, df["lon"].quantile(0.998) + 0.01)
    ax.set_ylim(df["lat"].quantile(0.002) - 0.01, df["lat"].quantile(0.998) + 0.01)
    ax.set_aspect(1 / np.cos(np.deg2rad(df["lat"].mean())))


def cell_centers(df, gx, gy, cell_m):
    lat0 = np.deg2rad(df["lat"].mean())
    lon_ref = np.deg2rad(df["lon"].mean())
    lat_ref = np.deg2rad(df["lat"].mean())
    x = (gx + 0.5) * cell_m
    y = (gy + 0.5) * cell_m
    lon = np.rad2deg(x / (RADIUS_M * np.cos(lat0)) + lon_ref)
    lat = np.rad2deg(y / RADIUS_M + lat_ref)
    return lon, lat


def run_dbscan(df):
    coords = np.deg2rad(df[["lat", "lon"]].to_numpy())
    labels = DBSCAN(
        eps=350 / RADIUS_M,
        min_samples=35,
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(coords)
    df = df.copy()
    df["cluster"] = labels
    clusters = (
        df[df["cluster"] != -1]
        .groupby("cluster")
        .agg(n=("INCIDENT_KEY", "size"), lat=("lat", "mean"), lon=("lon", "mean"), precinct=("PRECINCT", lambda s: s.mode().iat[0]), boro=("BORO", lambda s: s.mode().iat[0]))
        .sort_values("n", ascending=False)
    )
    return df, clusters


def plot_density(df):
    fig, ax = plt.subplots(figsize=(11, 12))
    hb = ax.hexbin(
        df["lon"],
        df["lat"],
        gridsize=115,
        mincnt=1,
        cmap="magma",
        norm=LogNorm(),
        linewidths=0,
        alpha=0.96,
    )
    add_dark_axes(ax, "NYC shooting density, 2006-2026 YTD")
    set_nyc_bounds(ax, df)
    cb = fig.colorbar(hb, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Incidents per hex, log scale", color="#c9d1d9")
    cb.ax.yaxis.set_tick_params(color="#c9d1d9")
    plt.setp(cb.ax.get_yticklabels(), color="#c9d1d9")
    ax.text(
        0.02,
        0.02,
        "Corrected coordinates: lat = source Longitude, lon = source Latitude",
        transform=ax.transAxes,
        color="#8b949e",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_density_hexbin.png", dpi=220)
    plt.close(fig)


def plot_dbscan(df, clusters):
    top = clusters.head(10)
    colors = [
        "#ff5a5f",
        "#00d4ff",
        "#ffe66d",
        "#7bed9f",
        "#c56cf0",
        "#ff9f43",
        "#54a0ff",
        "#f368e0",
        "#1dd1a1",
        "#ee5253",
    ]
    color_map = {cid: colors[i] for i, cid in enumerate(top.index)}

    fig, ax = plt.subplots(figsize=(11, 12))
    add_dark_axes(ax, "DBSCAN hotspot belts: 350m radius, min 35 incidents")
    set_nyc_bounds(ax, df)

    noise = df["cluster"].eq(-1)
    ax.scatter(df.loc[noise, "lon"], df.loc[noise, "lat"], s=2, c="#3d4852", alpha=0.24, linewidths=0)
    other = ~noise & ~df["cluster"].isin(top.index)
    ax.scatter(df.loc[other, "lon"], df.loc[other, "lat"], s=4, c="#7f8c8d", alpha=0.16, linewidths=0)

    for cid, row in top.iterrows():
        sub = df[df["cluster"].eq(cid)]
        ax.scatter(sub["lon"], sub["lat"], s=6, c=color_map[cid], alpha=0.44, linewidths=0, label=f"#{len(ax.collections)-1} {row.boro} Pct {row.precinct}: {int(row.n):,}")
        ax.scatter(row["lon"], row["lat"], s=95, c=color_map[cid], edgecolor="white", linewidth=1.2)

    for rank, (cid, row) in enumerate(top.head(6).iterrows(), start=1):
        ax.annotate(
            f"{rank}. {row.boro} P{int(row.precinct)}\n{int(row.n):,}",
            xy=(row["lon"], row["lat"]),
            xytext=(8, 8),
            textcoords="offset points",
            color="white",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="#161b22", ec=color_map[cid], alpha=0.88),
        )

    ax.text(
        0.02,
        0.02,
        f"{df['cluster'].ne(-1).mean() * 100:.1f}% of geocoded incidents fall inside DBSCAN clusters",
        transform=ax.transAxes,
        color="#f0f6fc",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", fc="#161b22", ec="#30363d", alpha=0.92),
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_dbscan_hotspot_belts.png", dpi=220)
    plt.close(fig)


def plot_shift(df):
    cell_m = 500
    df = df.copy()
    df["gx"] = np.floor(df["x_m"] / cell_m).astype(int)
    df["gy"] = np.floor(df["y_m"] / cell_m).astype(int)
    early = df[df["year"].between(2006, 2011)]
    recent = df[df["year"].between(2023, 2025)]

    early_rate = early.groupby(["gx", "gy"]).size() / 6
    recent_rate = recent.groupby(["gx", "gy"]).size() / 3
    cells = sorted(set(early_rate.index) | set(recent_rate.index))
    rows = []
    for gx, gy in cells:
        e = early_rate.get((gx, gy), 0)
        r = recent_rate.get((gx, gy), 0)
        if max(e, r) >= 2:
            lon, lat = cell_centers(df, gx, gy, cell_m)
            rows.append((gx, gy, lon, lat, e, r, r - e))
    shift = pd.DataFrame(rows, columns=["gx", "gy", "lon", "lat", "early_rate", "recent_rate", "diff"])

    fig, ax = plt.subplots(figsize=(11, 12))
    add_dark_axes(ax, "Where the map shifted: annualized recent vs early rate")
    set_nyc_bounds(ax, df)
    vmax = np.percentile(np.abs(shift["diff"]), 97)
    sc = ax.scatter(
        shift["lon"],
        shift["lat"],
        c=shift["diff"],
        s=np.clip(np.abs(shift["diff"]) * 18, 10, 260),
        cmap="coolwarm",
        norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax),
        alpha=0.86,
        linewidths=0.15,
        edgecolors="#0d1117",
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Recent annual rate minus 2006-2011 annual rate", color="#c9d1d9")
    plt.setp(cb.ax.get_yticklabels(), color="#c9d1d9")

    early_center = early[["lon", "lat"]].mean()
    recent_center = recent[["lon", "lat"]].mean()
    ax.annotate(
        "",
        xy=(recent_center["lon"], recent_center["lat"]),
        xytext=(early_center["lon"], early_center["lat"]),
        arrowprops=dict(arrowstyle="->", color="#ffe66d", lw=3),
    )
    ax.scatter([early_center["lon"], recent_center["lon"]], [early_center["lat"], recent_center["lat"]], c=["#58a6ff", "#ffe66d"], s=110, edgecolor="white", linewidth=1)
    ax.text(early_center["lon"], early_center["lat"] - 0.006, "2006-2011 center", color="#58a6ff", fontsize=9, ha="center")
    ax.text(recent_center["lon"], recent_center["lat"] + 0.006, "2023-2025 center", color="#ffe66d", fontsize=9, ha="center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_hotspot_shift_early_vs_recent.png", dpi=220)
    plt.close(fig)


def plot_emerging_fading(df):
    cell_m = 250
    df = df.copy()
    df["gx"] = np.floor(df["x_m"] / cell_m).astype(int)
    df["gy"] = np.floor(df["y_m"] / cell_m).astype(int)
    early = df[df["year"].between(2006, 2011)]
    recent = df[df["year"].between(2023, 2025)]
    early_count = early.groupby(["gx", "gy"]).size()
    recent_count = recent.groupby(["gx", "gy"]).size()

    rows = []
    for cell in sorted(set(early_count.index) | set(recent_count.index)):
        gx, gy = cell
        e = early_count.get(cell, 0)
        r = recent_count.get(cell, 0)
        if max(e, r) >= 8:
            lon, lat = cell_centers(df, gx, gy, cell_m)
            rows.append((gx, gy, lon, lat, e, r, r - e))
    cells = pd.DataFrame(rows, columns=["gx", "gy", "lon", "lat", "early", "recent", "diff"])
    emerging = cells.sort_values("diff", ascending=False).head(15)
    fading = cells.sort_values("diff", ascending=True).head(15)

    fig, ax = plt.subplots(figsize=(11, 12))
    add_dark_axes(ax, "Micro-hotspots that emerged or faded, 250m cells")
    set_nyc_bounds(ax, df)
    ax.scatter(df["lon"], df["lat"], s=1.2, c="#8b949e", alpha=0.12, linewidths=0)
    ax.scatter(fading["lon"], fading["lat"], s=np.clip(np.abs(fading["diff"]) * 12, 60, 300), marker="s", c="#58a6ff", alpha=0.78, label="Faded since 2006-2011")
    ax.scatter(emerging["lon"], emerging["lat"], s=np.clip(emerging["diff"] * 18, 80, 340), marker="^", c="#ff7b72", alpha=0.9, label="Emerging in 2023-2025")
    for _, row in emerging.head(7).iterrows():
        ax.annotate(
            f"{int(row.early)}->{int(row.recent)}",
            (row.lon, row.lat),
            xytext=(5, 4),
            textcoords="offset points",
            color="#ffebe9",
            fontsize=8,
        )
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_emerging_fading_250m_cells.png", dpi=220)
    plt.close(fig)


def plot_cluster_timeline(df, clusters):
    top = clusters.head(12).index
    annual = (
        df[df["cluster"].isin(top)]
        .pivot_table(index="cluster", columns="year", values="INCIDENT_KEY", aggfunc="count", fill_value=0)
        .reindex(top)
    )
    labels = []
    for cid in annual.index:
        row = clusters.loc[cid]
        labels.append(f"{row.boro} P{int(row.precinct)} ({int(row.n):,})")

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    im = ax.imshow(annual, aspect="auto", cmap="inferno")
    ax.set_title("Annual activity inside the top DBSCAN hotspot belts", loc="left", color="white", fontsize=18, fontweight="bold", pad=16)
    ax.set_yticks(range(len(labels)), labels=labels, color="#c9d1d9")
    ax.set_xticks(range(len(annual.columns)), labels=annual.columns, rotation=45, ha="right", color="#c9d1d9")
    ax.tick_params(colors="#c9d1d9")
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("Incidents", color="#c9d1d9")
    plt.setp(cb.ax.get_yticklabels(), color="#c9d1d9")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_dbscan_cluster_timeline.png", dpi=220)
    plt.close(fig)


def write_interactive(df, clusters):
    top = clusters.head(10)
    plot_df = df[df["cluster"].isin(top.index) | df["cluster"].eq(-1)].copy()
    plot_df["cluster_label"] = np.where(plot_df["cluster"].eq(-1), "Noise / outside belts", "Cluster " + plot_df["cluster"].astype(str))
    sample_noise = plot_df[plot_df["cluster"].eq(-1)].sample(min(2500, plot_df["cluster"].eq(-1).sum()), random_state=7)
    plot_df = pd.concat([plot_df[plot_df["cluster"].ne(-1)], sample_noise], ignore_index=True)

    fig = go.Figure()
    for cid, row in top.iterrows():
        sub = plot_df[plot_df["cluster"].eq(cid)]
        fig.add_trace(
            go.Scattermapbox(
                lon=sub["lon"],
                lat=sub["lat"],
                mode="markers",
                marker=dict(size=4, opacity=0.55),
                name=f"{row.boro} P{int(row.precinct)}: {int(row.n):,}",
                text=sub["date"].dt.strftime("%Y-%m-%d") + "<br>" + sub["BORO"].astype(str) + " Pct " + sub["PRECINCT"].astype(str) + "<br>" + sub["loc_class"].astype(str),
                hoverinfo="text",
            )
        )
    noise = plot_df[plot_df["cluster"].eq(-1)]
    fig.add_trace(go.Scattermapbox(lon=noise["lon"], lat=noise["lat"], mode="markers", marker=dict(size=3, color="gray", opacity=0.22), name="outside top belts"))
    fig.update_layout(
        title="Interactive DBSCAN Hotspot Belts, NYC Shootings",
        template="plotly_dark",
        width=1050,
        height=900,
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lon=df["lon"].mean(), lat=df["lat"].mean()),
            zoom=9.5,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(OUTPUT_DIR / "interactive_dbscan_hotspots.html", include_plotlyjs="cdn")


def make_animation(df):
    width, height = 1000, 1100
    years = sorted(df["year"].unique())
    bounds = {
        "lon_min": df["lon"].quantile(0.002) - 0.01,
        "lon_max": df["lon"].quantile(0.998) + 0.01,
        "lat_min": df["lat"].quantile(0.002) - 0.01,
        "lat_max": df["lat"].quantile(0.998) + 0.01,
    }
    frames = []
    tmp_dir = OUTPUT_DIR / "animation_frames"
    tmp_dir.mkdir(exist_ok=True)
    colors = {
        "BROOKLYN": "#ff5a5f",
        "BRONX": "#00d4ff",
        "MANHATTAN": "#ffe66d",
        "QUEENS": "#7bed9f",
        "STATEN ISLAND": "#c56cf0",
    }

    for year in years:
        fig, ax = plt.subplots(figsize=(10, 11))
        add_dark_axes(ax, f"Shooting incidents through {year}")
        ax.set_xlim(bounds["lon_min"], bounds["lon_max"])
        ax.set_ylim(bounds["lat_min"], bounds["lat_max"])
        ax.set_aspect(1 / np.cos(np.deg2rad(df["lat"].mean())))

        history = df[df["year"] < year]
        current = df[df["year"] == year]
        ax.scatter(history["lon"], history["lat"], s=2, c="#56606a", alpha=0.09, linewidths=0)
        for boro, sub in current.groupby("BORO"):
            ax.scatter(sub["lon"], sub["lat"], s=9, c=colors.get(boro, "white"), alpha=0.78, linewidths=0, label=boro)
        ax.text(
            0.03,
            0.04,
            f"{len(current):,} incidents in {year} | {len(df[df['year'] <= year]):,} cumulative",
            transform=ax.transAxes,
            color="#f0f6fc",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.35", fc="#161b22", ec="#30363d", alpha=0.92),
        )
        ax.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc", markerscale=1.8)
        frame_path = tmp_dir / f"frame_{year}.png"
        fig.tight_layout()
        fig.savefig(frame_path, dpi=130)
        plt.close(fig)
        frames.append(imageio.imread(frame_path))

    gif_path = OUTPUT_DIR / "shootings_hotspot_animation_2006_2026.gif"
    imageio.mimsave(gif_path, frames, duration=0.55, loop=0)
    return gif_path


def write_summary(df, clusters):
    cell_m = 250
    tmp = df.copy()
    tmp["gx"] = np.floor(tmp["x_m"] / cell_m).astype(int)
    tmp["gy"] = np.floor(tmp["y_m"] / cell_m).astype(int)
    counts = tmp.groupby(["gx", "gy"]).size().sort_values(ascending=False)
    top_5pct = max(1, int(np.ceil(len(counts) * 0.05)))
    top_5_share = counts.head(top_5pct).sum() / len(tmp) * 100
    clustered_share = df["cluster"].ne(-1).mean() * 100
    top3_share = clusters.head(3)["n"].sum() / len(tmp) * 100
    early = df[df["year"].between(2006, 2011)]
    recent = df[df["year"].between(2023, 2025)]
    km_shift = np.sqrt(((recent[["x_m", "y_m"]].mean() - early[["x_m", "y_m"]].mean()) ** 2).sum()) / 1000

    lines = [
        "# Spatial Hotspot Plot Notes",
        "",
        f"- Geocoded incidents used: {len(df):,}. Coordinate fields were corrected because source `Latitude` contains longitude-like values and source `Longitude` contains latitude-like values.",
        f"- DBSCAN settings: 350 meter radius, minimum 35 incidents.",
        f"- {clustered_share:.1f}% of geocoded incidents are inside DBSCAN clusters.",
        f"- The top 3 DBSCAN belts contain {clusters.head(3)['n'].sum():,} incidents, or {top3_share:.1f}% of all geocoded incidents.",
        f"- The top 5% of occupied 250m grid cells contain {top_5_share:.1f}% of all geocoded incidents.",
        f"- The incident centroid moved about {km_shift:.1f} km from the 2006-2011 era to the 2023-2025 era.",
        "",
        "## Files",
        "",
        "- `01_density_hexbin.png`: all-incident spatial density.",
        "- `02_dbscan_hotspot_belts.png`: top DBSCAN hotspot belts.",
        "- `03_hotspot_shift_early_vs_recent.png`: annualized 500m-cell change, 2023-2025 vs 2006-2011.",
        "- `04_emerging_fading_250m_cells.png`: most changed 250m cells.",
        "- `05_dbscan_cluster_timeline.png`: annual activity by major DBSCAN belt.",
        "- `interactive_dbscan_hotspots.html`: interactive Plotly hotspot explorer.",
        "- `shootings_hotspot_animation_2006_2026.gif`: year-by-year incident animation.",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines))


def main():
    df = load_points()
    df, clusters = run_dbscan(df)
    plot_density(df)
    plot_dbscan(df, clusters)
    plot_shift(df)
    plot_emerging_fading(df)
    plot_cluster_timeline(df, clusters)
    write_interactive(df, clusters)
    gif_path = make_animation(df)
    write_summary(df, clusters)
    print(f"Wrote spatial outputs to {OUTPUT_DIR.resolve()}")
    print(f"Animation: {gif_path}")


if __name__ == "__main__":
    main()
