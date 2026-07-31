import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

RADIUS_M = 6_371_008.8
DATA_PATH = Path("shootings_nyc.csv")

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["lon"] = df["Latitude"]
    df["lat"] = df["Longitude"]
    df = df[df["lon"].between(-75, -73) & df["lat"].between(40, 41)].copy()
    return df

def find_optimal_radius(df, min_samples=35):
    coords = np.deg2rad(df[["lat", "lon"]].to_numpy())
    nn = NearestNeighbors(n_neighbors=min_samples, metric="haversine", algorithm="ball_tree")
    nn.fit(coords)
    distances, _ = nn.kneighbors(coords)
    
    # Distance to the k-th nearest neighbor
    k_distances = distances[:, -1]
    k_distances = np.sort(k_distances)
    
    # Convert from radians to meters
    k_distances_m = k_distances * RADIUS_M
    
    # Find the 'elbow' point using the maximum curvature method (simple line distance)
    p1 = np.array([0, k_distances_m[0]])
    p2 = np.array([len(k_distances_m)-1, k_distances_m[-1]])
    
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    
    max_dist = 0
    elbow_idx = 0
    
    for i, d in enumerate(k_distances_m):
        p = np.array([i, d])
        vec_to_p1 = p - p1
        proj_length = np.dot(vec_to_p1, line_vec_norm)
        proj_point = p1 + proj_length * line_vec_norm
        dist_to_line = np.linalg.norm(p - proj_point)
        
        if dist_to_line > max_dist:
            max_dist = dist_to_line
            elbow_idx = i
            
    optimal_radius = k_distances_m[elbow_idx]
    print(f"Mathematical Optimal Radius (Elbow Method): {optimal_radius:.1f} meters")
    
    # Save to a text file to read from the next script
    with open("outputs/optimal_radius.txt", "w") as f:
        f.write(str(optimal_radius))

if __name__ == "__main__":
    df = load_data()
    find_optimal_radius(df)
