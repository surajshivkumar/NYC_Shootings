# Spatial Hotspot Plot Notes

- Geocoded incidents used: 23,988. Coordinate fields were corrected because source `Latitude` contains longitude-like values and source `Longitude` contains latitude-like values.
- DBSCAN settings: 350 meter radius, minimum 35 incidents.
- 80.9% of geocoded incidents are inside DBSCAN clusters.
- The top 3 DBSCAN belts contain 14,802 incidents, or 61.7% of all geocoded incidents.
- The top 5% of occupied 250m grid cells contain 25.1% of all geocoded incidents.
- The incident centroid moved about 2.1 km from the 2006-2011 era to the 2023-2025 era.

## Files

- `01_density_hexbin.png`: all-incident spatial density.
- `02_dbscan_hotspot_belts.png`: top DBSCAN hotspot belts.
- `03_hotspot_shift_early_vs_recent.png`: annualized 500m-cell change, 2023-2025 vs 2006-2011.
- `04_emerging_fading_250m_cells.png`: most changed 250m cells.
- `05_dbscan_cluster_timeline.png`: annual activity by major DBSCAN belt.
- `interactive_dbscan_hotspots.html`: interactive Plotly hotspot explorer.
- `shootings_hotspot_animation_2006_2026.gif`: year-by-year incident animation.