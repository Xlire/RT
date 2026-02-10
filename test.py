import numpy as np
import open3d as o3d

# Load the file
data = np.loadtxt("test_pointcloud.txt")

# Split columns
points = data[:, :3]      # X Y Z
colors = data[:, 3:6] / 255.0  # Normalize RGB to [0,1]
timestamps = data[:, 6]

drift_rate = np.array([5, 0, 0])  # 1 cm per second
drift = timestamps[:, None] * drift_rate
drifted_points = points + drift


# Create clean cloud (black)
pcd_clean = o3d.geometry.PointCloud()
pcd_clean.points = o3d.utility.Vector3dVector(points)


# Create drifted cloud (red)
pcd_drift = o3d.geometry.PointCloud()
pcd_drift.points = o3d.utility.Vector3dVector(drifted_points)


# pcd_clean = pcd_clean.voxel_down_sample(0.5)
# pcd_drift = pcd_drift.voxel_down_sample(0.5)

# Save to txt
def save_txt_with_time(filename, pcd, timestamps):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    with open(filename, "w") as f:
        for (p, c, t) in zip(points, colors, timestamps):
            r, g, b = (c * 255).astype(int)
            f.write(f"{p[0]} {p[1]} {p[2]} {r} {g} {b} {t}\n")


save_txt_with_time("clean_output.txt", pcd_clean, timestamps)
save_txt_with_time("drifted_output.txt", pcd_drift, timestamps)

# Visualize
pcd_clean.colors = o3d.utility.Vector3dVector(
    np.zeros_like(points)  #black
)
pcd_drift.colors = o3d.utility.Vector3dVector(
    np.tile([1, 0, 0], (points.shape[0], 1))  # red
)
o3d.visualization.draw_geometries([pcd_clean, pcd_drift])






