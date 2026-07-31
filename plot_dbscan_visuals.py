"""
Generate DBSCAN visualization plots:
1. K-Distance Elbow curve (how we found 979m)
2. Per-borough cluster maps for a representative year
3. All-city cluster view
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

OUTPUT_DIR = Path("outputs/spatial")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
cache_dir = OUTPUT_DIR / ".cache"
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir.resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

RADIUS_M = 6_371_008.8
DATA_PATH = Path("shootings_nyc.csv")

BORO_COLORS = {
    "BRONX": "#00d4ff",
    "BROOKLYN": "#ff5a5f",
    "MANHATTAN": "#ffe66d",
    "QUEENS": "#7bed9f",
    "STATEN ISLAND": "#c56cf0",
}

CLUSTER_COLORS = [
    "#ff5a5f", "#00d4ff", "#ffe66d", "#7bed9f", "#c56cf0",
    "#ff9f43", "#54a0ff", "#f368e0", "#1dd1a1", "#ee5253",
    "#feca57", "#48dbfb", "#ff6b6b", "#a29bfe", "#fd79a8",
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["OCCUR_DATE"])
    df["year"] = df["date"].dt.year
    df["lon"] = df["Latitude"]
    df["lat"] = df["Longitude"]
    df = df[df["lon"].between(-75, -73) & df["lat"].between(40, 41)].copy()
    df = df[df["BORO"].isin(BORO_COLORS)].copy()
    return df


def dark_style(ax, title):
    ax.set_facecolor("#0d1117")
    ax.figure.set_facecolor("#0d1117")
    ax.set_title(title, loc="left", color="white", fontsize=16, fontweight="bold", pad=12)
    ax.tick_params(colors="#8b949e", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(color="#21262d", alpha=0.5, linewidth=0.5)


def plot_elbow(df):
    """Plot 1: The K-Distance Elbow curve showing how we found 979m."""
    print("Computing K-distances (this takes a moment)...")
    coords = np.deg2rad(df[["lat", "lon"]].to_numpy())
    nn = NearestNeighbors(n_neighbors=35, metric="haversine", algorithm="ball_tree")
    nn.fit(coords)
    distances, _ = nn.kneighbors(coords)

    k_distances = distances[:, -1]
    k_distances = np.sort(k_distances)
    k_distances_m = k_distances * RADIUS_M

    # Find elbow
    p1 = np.array([0, k_distances_m[0]])
    p2 = np.array([len(k_distances_m) - 1, k_distances_m[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    max_dist = 0
    elbow_idx = 0
    for i, d in enumerate(k_distances_m):
        p = np.array([i, d])
        vec = p - p1
        proj = np.dot(vec, line_vec_norm)
        proj_pt = p1 + proj * line_vec_norm
        dist = np.linalg.norm(p - proj_pt)
        if dist > max_dist:
            max_dist = dist
            elbow_idx = i

    optimal_radius = k_distances_m[elbow_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    dark_style(ax, "K-Distance Plot: Finding the Optimal DBSCAN Radius")

    # Plot the curve
    x_vals = np.arange(len(k_distances_m))
    ax.plot(x_vals, k_distances_m, color="#58a6ff", linewidth=1.5, alpha=0.9)

    # Fill under curve
    ax.fill_between(x_vals, k_distances_m, alpha=0.08, color="#58a6ff")

    # Mark the elbow
    ax.axhline(y=optimal_radius, color="#ff5a5f", linestyle="--", linewidth=2, alpha=0.8)
    ax.plot(elbow_idx, optimal_radius, "o", color="#ff5a5f", markersize=14, zorder=5)
    ax.annotate(
        f"  Elbow = {optimal_radius:.0f}m\n  (Optimal Radius)",
        xy=(elbow_idx, optimal_radius),
        xytext=(elbow_idx + len(k_distances_m) * 0.05, optimal_radius + 800),
        color="#ff5a5f",
        fontsize=13,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#ff5a5f", lw=2),
    )

    # Annotate zones
    ax.annotate(
        "← Dense hotspot zone\n     (points have many neighbors nearby)",
        xy=(len(k_distances_m) * 0.15, k_distances_m[int(len(k_distances_m) * 0.15)]),
        color="#7bed9f",
        fontsize=11,
        fontweight="bold",
    )
    ax.annotate(
        "Noise / isolated incidents →\n(too far from other points to form clusters)",
        xy=(len(k_distances_m) * 0.75, k_distances_m[int(len(k_distances_m) * 0.75)] + 200),
        color="#8b949e",
        fontsize=11,
        ha="center",
    )

    ax.set_xlabel("Shooting incidents (sorted by distance to 35th nearest neighbor)", color="#8b949e", fontsize=11)
    ax.set_ylabel("Distance to 35th nearest neighbor (meters)", color="#8b949e", fontsize=11)
    ax.set_ylim(0, min(k_distances_m[-1], 8000))

    fig.tight_layout()
    out = OUTPUT_DIR / "dbscan_elbow_curve.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")
    return optimal_radius


def plot_clusters_year(df, year, eps_m):
    """Plot 2: Show DBSCAN clusters for a single year on a scatter plot."""
    sub = df[df["year"] == year].copy()
    coords = np.deg2rad(sub[["lat", "lon"]].to_numpy())

    labels = DBSCAN(
        eps=eps_m / RADIUS_M,
        min_samples=25,
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(coords)

    sub["cluster"] = labels
    n_clusters = len(set(labels) - {-1})

    fig, ax = plt.subplots(figsize=(12, 13))
    dark_style(ax, f"DBSCAN Clustering — {year}  |  {n_clusters} clusters found  |  radius = {eps_m:.0f}m")

    # Plot noise first (grey, tiny)
    noise = sub[sub["cluster"] == -1]
    ax.scatter(noise["lon"], noise["lat"], s=4, c="#30363d", alpha=0.4, label=f"Noise ({len(noise):,})", zorder=1)

    # Plot each cluster
    cluster_ids = sorted(set(labels) - {-1}, key=lambda c: -(sub["cluster"] == c).sum())
    for rank, cid in enumerate(cluster_ids[:15]):
        mask = sub["cluster"] == cid
        cluster_pts = sub[mask]
        color = CLUSTER_COLORS[rank % len(CLUSTER_COLORS)]
        size = len(cluster_pts)
        ax.scatter(
            cluster_pts["lon"], cluster_pts["lat"],
            s=18, c=color, alpha=0.75, zorder=2,
            label=f"Cluster {rank+1}: {size} incidents",
        )
        # Draw centroid
        clat, clon = cluster_pts["lat"].mean(), cluster_pts["lon"].mean()
        ax.plot(clon, clat, "*", color=color, markersize=18, markeredgecolor="white", markeredgewidth=1.5, zorder=4)

        # Draw approximate radius circle
        r_deg_lon = (eps_m / RADIUS_M) / np.cos(np.radians(40.7)) * (180 / np.pi)
        r_deg_lat = (eps_m / RADIUS_M) * (180 / np.pi)
        circle = matplotlib.patches.Ellipse(
            (clon, clat), width=r_deg_lon * 2, height=r_deg_lat * 2,
            fill=False, edgecolor=color, linewidth=1.5, linestyle="--", alpha=0.5, zorder=3,
        )
        ax.add_patch(circle)

    ax.set_xlim(sub["lon"].quantile(0.005) - 0.01, sub["lon"].quantile(0.995) + 0.01)
    ax.set_ylim(sub["lat"].quantile(0.005) - 0.01, sub["lat"].quantile(0.995) + 0.01)
    ax.set_aspect(1 / np.cos(np.radians(40.7)))
    ax.set_xlabel("Longitude", color="#8b949e")
    ax.set_ylabel("Latitude", color="#8b949e")

    legend = ax.legend(
        loc="upper right", fontsize=9, facecolor="#161b22", edgecolor="#30363d",
        labelcolor="#c9d1d9", framealpha=0.9,
    )

    fig.tight_layout()
    out = OUTPUT_DIR / f"dbscan_clusters_{year}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def plot_before_after_covid(df, eps_m):
    """Plot 3: Side-by-side 2019 vs 2020 showing the COVID explosion."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 12))

    for ax, year in zip(axes, [2019, 2020]):
        sub = df[df["year"] == year].copy()
        coords = np.deg2rad(sub[["lat", "lon"]].to_numpy())
        labels = DBSCAN(eps=eps_m / RADIUS_M, min_samples=20, metric="haversine", algorithm="ball_tree").fit_predict(coords)
        sub["cluster"] = labels
        n_clusters = len(set(labels) - {-1})
        n_clustered = (labels != -1).sum()

        dark_style(ax, f"{year}  —  {n_clusters} clusters, {n_clustered:,} clustered incidents")

        noise = sub[sub["cluster"] == -1]
        ax.scatter(noise["lon"], noise["lat"], s=5, c="#30363d", alpha=0.35, zorder=1)

        cluster_ids = sorted(set(labels) - {-1}, key=lambda c: -(sub["cluster"] == c).sum())
        for rank, cid in enumerate(cluster_ids[:12]):
            mask = sub["cluster"] == cid
            cluster_pts = sub[mask]
            color = CLUSTER_COLORS[rank % len(CLUSTER_COLORS)]
            ax.scatter(cluster_pts["lon"], cluster_pts["lat"], s=20, c=color, alpha=0.7, zorder=2)
            clat, clon = cluster_pts["lat"].mean(), cluster_pts["lon"].mean()
            ax.plot(clon, clat, "*", color=color, markersize=16, markeredgecolor="white", markeredgewidth=1.2, zorder=4)

        ax.set_xlim(-74.05, -73.7)
        ax.set_ylim(40.55, 40.92)
        ax.set_aspect(1 / np.cos(np.radians(40.7)))
        ax.set_xlabel("Longitude", color="#8b949e")
        ax.set_ylabel("Latitude", color="#8b949e")

    fig.suptitle(
        "The COVID Explosion: 2019 vs 2020",
        color="white", fontsize=22, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUTPUT_DIR / "dbscan_covid_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def plot_eli5_how_it_works(df, eps_m):
    """Plot 4: ELI5 step-by-step visual of how DBSCAN works on a small patch."""
    # Zoom into a small area (Brownsville, Brooklyn) for clarity
    sub = df[
        (df["year"] == 2020)
        & (df["lon"].between(-73.935, -73.905))
        & (df["lat"].between(40.66, 40.685))
    ].copy()

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.figure.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.set_xlim(-73.936, -73.904)
        ax.set_ylim(40.659, 40.686)
        ax.set_aspect(1 / np.cos(np.radians(40.7)))

    # Step 1: Raw points
    ax1 = axes[0]
    ax1.set_title("Step 1: Raw incidents", color="white", fontsize=14, fontweight="bold", pad=10)
    ax1.scatter(sub["lon"], sub["lat"], s=30, c="#58a6ff", alpha=0.6, zorder=2)
    ax1.text(0.5, 0.02, f"{len(sub)} shooting incidents in Brownsville, 2020",
             transform=ax1.transAxes, color="#8b949e", fontsize=10, ha="center")

    # Step 2: Draw radius circles around a few seed points
    ax2 = axes[1]
    ax2.set_title(f"Step 2: Draw {eps_m:.0f}m radius circles", color="white", fontsize=14, fontweight="bold", pad=10)
    ax2.scatter(sub["lon"], sub["lat"], s=30, c="#58a6ff", alpha=0.4, zorder=2)
    
    # Pick 5 seed points
    seeds = sub.sample(5, random_state=42)
    r_deg_lon = (eps_m / RADIUS_M) / np.cos(np.radians(40.7)) * (180 / np.pi)
    r_deg_lat = (eps_m / RADIUS_M) * (180 / np.pi)
    for _, seed in seeds.iterrows():
        circle = matplotlib.patches.Ellipse(
            (seed["lon"], seed["lat"]), width=r_deg_lon * 2, height=r_deg_lat * 2,
            fill=False, edgecolor="#ff5a5f", linewidth=2, linestyle="--", alpha=0.7, zorder=3,
        )
        ax2.add_patch(circle)
        ax2.plot(seed["lon"], seed["lat"], "o", color="#ff5a5f", markersize=10, zorder=4)
    ax2.text(0.5, 0.02, "Each point checks: are there ≥35 neighbors\nwithin my 979m radius?",
             transform=ax2.transAxes, color="#8b949e", fontsize=10, ha="center")

    # Step 3: Final clusters
    ax3 = axes[2]
    ax3.set_title("Step 3: Clusters found!", color="white", fontsize=14, fontweight="bold", pad=10)
    coords = np.deg2rad(sub[["lat", "lon"]].to_numpy())
    labels = DBSCAN(eps=eps_m / RADIUS_M, min_samples=15, metric="haversine", algorithm="ball_tree").fit_predict(coords)
    sub = sub.copy()
    sub["cluster"] = labels
    
    noise = sub[sub["cluster"] == -1]
    ax3.scatter(noise["lon"], noise["lat"], s=25, c="#30363d", alpha=0.5, marker="x", zorder=1, label=f"Noise ({len(noise)})")
    
    cluster_ids = sorted(set(labels) - {-1}, key=lambda c: -(sub["cluster"] == c).sum())
    for rank, cid in enumerate(cluster_ids[:5]):
        cluster_pts = sub[sub["cluster"] == cid]
        color = CLUSTER_COLORS[rank % len(CLUSTER_COLORS)]
        ax3.scatter(cluster_pts["lon"], cluster_pts["lat"], s=35, c=color, alpha=0.75, zorder=2,
                   label=f"Cluster {rank+1} ({len(cluster_pts)} pts)")
        clat, clon = cluster_pts["lat"].mean(), cluster_pts["lon"].mean()
        ax3.plot(clon, clat, "*", color="white", markersize=16, zorder=4)

    legend = ax3.legend(loc="lower right", fontsize=9, facecolor="#161b22", edgecolor="#30363d",
                        labelcolor="#c9d1d9", framealpha=0.9)

    fig.tight_layout()
    out = OUTPUT_DIR / "dbscan_eli5_steps.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def main():
    df = load_data()
    
    print("=== Plot 1: K-Distance Elbow ===")
    eps = plot_elbow(df)
    
    print("=== Plot 2: DBSCAN Clusters 2020 ===")
    plot_clusters_year(df, 2020, eps)
    
    print("=== Plot 3: DBSCAN Clusters 2006 ===")
    plot_clusters_year(df, 2006, eps)
    
    print("=== Plot 4: COVID Comparison ===")
    plot_before_after_covid(df, eps)
    
    print("=== Plot 5: ELI5 Step-by-Step ===")
    plot_eli5_how_it_works(df, eps)
    
    print("\nAll done!")


if __name__ == "__main__":
    main()
