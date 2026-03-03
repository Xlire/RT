import numpy as np
import open3d as o3d

# ---------------------------------------------------------
# 1. Load Unity point cloud
# ---------------------------------------------------------
data = np.loadtxt("test_pointcloud.txt")

points = data[:, :3]                 # X Y Z
colors = data[:, 3:6] / 255.0        # Normalize RGB to [0,1]
timestamps = data[:, 6]              # Time of each point


# ---------------------------------------------------------
# 2. Simulate object-level drift (rigid transform on a difine region)
# ---------------------------------------------------------

# 2.1 Define a region (example: points with x > 0 and z < 1)
region_mask = (points[:, 0] > 0) & (points[:, 2] < 10)

# 2.2 Choose a time threshold t0
t0 = np.percentile(timestamps, 50)   # drift starts halfway through the scan
time_mask = timestamps >= t0

# Combined mask: points in region AND after t0
drift_mask = region_mask & time_mask

# 2.3 Generate a random rotation rigid transform (R, t)
def random_rotation_matrix(max_angle_deg=10):
    angle = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    ux, uy, uz = axis
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([
        [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),    uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)]
    ])
    return R

# R = random_rotation_matrix(max_angle_deg=25)  # small rotation
theta = np.radians(45)   # rotate 45 degrees to the left
c, s = np.cos(theta), np.sin(theta)

R = np.array([
    [ c, 0,  s],
    [ 0, 1,  0],
    [-s, 0,  c]
])

drifted_points = points.copy()

# Select the points that should rotate
pts = points[drift_mask]

# 1. Find the center of the object
center = pts.mean(axis=0)

# 2. Move object to origin
pts_centered = pts - center

# 3. Rotate around Y-axis
pts_rotated = (R @ pts_centered.T).T

# 4. Move back to original location
pts_rotated += center

# Save back into the drifted cloud
drifted_points[drift_mask] = pts_rotated


direction = np.random.randn(3)
direction /= np.linalg.norm(direction)

t = 0 * direction  # drift magnitude ~30 cm

# 2.4 Apply rigid transform only to selected points
# drifted_points = points.copy()
# #Look into this 
# drifted_points[drift_mask] = (R @ points[drift_mask].T).T + t

# print(drifted_points)


# ---------------------------------------------------------
# 3. Create clean and drifted point clouds
# ---------------------------------------------------------
pcd_clean = o3d.geometry.PointCloud()
pcd_clean.points = o3d.utility.Vector3dVector(points)
pcd_clean.colors = o3d.utility.Vector3dVector(colors)   # IMPORTANT

pcd_drift = o3d.geometry.PointCloud()
pcd_drift.points = o3d.utility.Vector3dVector(drifted_points)
pcd_drift.colors = o3d.utility.Vector3dVector(colors)   # IMPORTANT

# ---------------------------------------------------------
# 4. Save to TXT (preserving timestamps)
# ---------------------------------------------------------
def save_txt_with_time(filename, pcd, timestamps):
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)

    with open(filename, "w") as f:
        for (p, c, t) in zip(pts, cols, timestamps):
            r, g, b = (c * 255).astype(int)
            f.write(f"{p[0]} {p[1]} {p[2]} {r} {g} {b} {t}\n")


save_txt_with_time("clean_output.txt", pcd_clean, timestamps)
save_txt_with_time("drifted_output.txt", pcd_drift, timestamps)

# ---------------------------------------------------------
# 5. Visualization (black = clean, red = drifted)
# ---------------------------------------------------------
pcd_clean.colors = o3d.utility.Vector3dVector(
    np.zeros_like(points)  # black
)

pcd_drift.colors = o3d.utility.Vector3dVector(
    np.tile([1, 0, 0], (points.shape[0], 1))  # red
)

o3d.visualization.draw_geometries([pcd_clean, pcd_drift])