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
region_mask = (points[:, 0] > 0) & (points[:, 2] < 1)

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

R = random_rotation_matrix(max_angle_deg=5)  # small rotation
direction = np.random.randn(3)
direction /= np.linalg.norm(direction)
t = 0.3 * direction  # drift magnitude ~30 cm

# 2.4 Apply rigid transform only to selected points
drifted_points = points.copy()
#Look into this 
drifted_points[drift_mask] = (R @ points[drift_mask].T).T + t

print(drifted_points)