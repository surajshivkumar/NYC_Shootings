import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd


DATA_PATH = Path("shootings_nyc.csv")
OUTPUT_DIR = Path("outputs/spatial")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cache_dir = OUTPUT_DIR / ".cache"
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RADIUS_M = 6_371_008.8


def load_points():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["OCCUR_DATE"])
    df["year"] = df["date"].dt.year

    # Source coordinate values are swapped in this file.
    df["lon"] = df["Latitude"]
    df["lat"] = df["Longitude"]
    df = df[df["lon"].between(-75, -73) & df["lat"].between(40, 41)].copy()

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


def centroid(df):
    x = df["x_m"].mean()
    y = df["y_m"].mean()
    lon, lat = xy_to_lonlat(df, x, y)
    return pd.Series({"x_m": x, "y_m": y, "lon": lon, "lat": lat, "n": len(df)})


def make_centers(df):
    annual = df.groupby("year").apply(centroid, include_groups=False).reset_index()
    cumulative_rows = []
    for year in annual["year"]:
        row = centroid(df[df["year"] <= year])
        row["year"] = year
        cumulative_rows.append(row)
    cumulative = pd.DataFrame(cumulative_rows)
    return annual, cumulative


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


def set_bounds(ax, df):
    ax.set_xlim(df["lon"].quantile(0.002) - 0.01, df["lon"].quantile(0.998) + 0.01)
    ax.set_ylim(df["lat"].quantile(0.002) - 0.01, df["lat"].quantile(0.998) + 0.01)
    ax.set_aspect(1 / np.cos(np.deg2rad(df["lat"].mean())))


def make_static_plot(df, annual, cumulative):
    fig, ax = plt.subplots(figsize=(10, 11))
    add_dark_axes(ax, "Annual and cumulative shooting centroid")
    set_bounds(ax, df)
    ax.scatter(df["lon"], df["lat"], s=1.4, c="#8b949e", alpha=0.10, linewidths=0)
    ax.plot(annual["lon"], annual["lat"], color="#ff7b72", linewidth=1.6, alpha=0.85, label="Annual center")
    ax.scatter(annual["lon"], annual["lat"], c=annual["year"], cmap="plasma", s=52, edgecolor="white", linewidth=0.45)
    ax.plot(cumulative["lon"], cumulative["lat"], color="#58a6ff", linewidth=3, label="Cumulative center")
    ax.scatter(cumulative.iloc[0]["lon"], cumulative.iloc[0]["lat"], s=120, c="#58a6ff", edgecolor="white", label="Start")
    ax.scatter(cumulative.iloc[-1]["lon"], cumulative.iloc[-1]["lat"], s=140, c="#ffe66d", edgecolor="white", label="Latest")
    ax.annotate(
        "",
        xy=(cumulative.iloc[-1]["lon"], cumulative.iloc[-1]["lat"]),
        xytext=(cumulative.iloc[0]["lon"], cumulative.iloc[0]["lat"]),
        arrowprops=dict(arrowstyle="->", color="#ffe66d", lw=2.5),
    )
    ax.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_centroid_path.png", dpi=220)
    plt.close(fig)


def make_animation(df, annual, cumulative):
    frames = []
    frame_dir = OUTPUT_DIR / "center_frames"
    frame_dir.mkdir(exist_ok=True)
    years = annual["year"].tolist()
    colors = {
        "BROOKLYN": "#ff5a5f",
        "BRONX": "#00d4ff",
        "MANHATTAN": "#ffe66d",
        "QUEENS": "#7bed9f",
        "STATEN ISLAND": "#c56cf0",
    }

    for year in years:
        fig, ax = plt.subplots(figsize=(10, 11))
        add_dark_axes(ax, f"Center of shooting incidents: {year}")
        set_bounds(ax, df)

        past = df[df["year"] < year]
        current = df[df["year"] == year]
        ax.scatter(past["lon"], past["lat"], s=1.2, c="#56606a", alpha=0.055, linewidths=0)
        for boro, sub in current.groupby("BORO"):
            ax.scatter(sub["lon"], sub["lat"], s=8, c=colors.get(boro, "white"), alpha=0.55, linewidths=0)

        shown_annual = annual[annual["year"] <= year]
        shown_cumulative = cumulative[cumulative["year"] <= year]
        ax.plot(shown_annual["lon"], shown_annual["lat"], color="#ff7b72", linewidth=1.5, alpha=0.8)
        ax.scatter(shown_annual["lon"], shown_annual["lat"], s=38, c="#ff7b72", edgecolor="white", linewidth=0.4, alpha=0.9, label="Annual center")
        ax.plot(shown_cumulative["lon"], shown_cumulative["lat"], color="#58a6ff", linewidth=3.0, alpha=0.95, label="Cumulative center")
        latest_annual = shown_annual.iloc[-1]
        latest_cum = shown_cumulative.iloc[-1]
        ax.scatter(latest_annual["lon"], latest_annual["lat"], s=190, c="#ff7b72", edgecolor="white", linewidth=1.4)
        ax.scatter(latest_cum["lon"], latest_cum["lat"], s=210, c="#58a6ff", edgecolor="white", linewidth=1.4)
        ax.text(
            0.03,
            0.045,
            f"Annual center = mean x/y of {int(latest_annual.n):,} incidents in {year}\n"
            f"Cumulative center = mean x/y of all incidents through {year}",
            transform=ax.transAxes,
            color="#f0f6fc",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", fc="#161b22", ec="#30363d", alpha=0.92),
        )
        ax.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc")
        fig.tight_layout()
        frame_path = frame_dir / f"center_{year}.png"
        fig.savefig(frame_path, dpi=130)
        plt.close(fig)
        frames.append(imageio.imread(frame_path))

    start_year = int(years[0])
    end_year = int(years[-1])
    gif_path = OUTPUT_DIR / f"shooting_centroid_shift_{start_year}_{end_year}.gif"
    imageio.mimsave(gif_path, frames, duration=0.65, loop=0)
    return gif_path


def main():
    df = load_points()
    annual, cumulative = make_centers(df)
    make_static_plot(df, annual, cumulative)
    gif_path = make_animation(df, annual, cumulative)
    annual[["year", "lat", "lon", "n"]].to_csv(OUTPUT_DIR / "annual_centroids.csv", index=False)
    cumulative[["year", "lat", "lon", "n"]].to_csv(OUTPUT_DIR / "cumulative_centroids.csv", index=False)
    print(f"Wrote {OUTPUT_DIR / '06_centroid_path.png'}")
    print(f"Wrote {gif_path}")
    print(f"Wrote centroid CSVs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
