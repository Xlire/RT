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
# 2. Simulate drift
# ---------------------------------------------------------
drift_rate = np.array([0.2, 0, 0])     # Drift 5 units per second along X
drift = timestamps[:, None] * drift_rate
drifted_points = points + drift


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

threshold = 0.5  # max correspondence distance
trans_init = np.eye(4)

reg = o3d.pipelines.registration.registration_icp(
    pcd_drift, pcd_clean, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

print("ICP fitness:", reg.fitness)
print("ICP RMSE:", reg.inlier_rmse)
print("Estimated correction:\n", reg.transformation)

error = np.linalg.norm(points - drifted_points, axis=1)
print("Mean drift:", error.mean())
print("Max drift:", error.max())

T = reg.transformation  # 4x4 matrix you printed

# Apply correction to the drifted cloud (in-place)
pcd_drift_corrected = pcd_drift.transform(T)

o3d.visualization.draw_geometries([pcd_clean, pcd_drift_corrected])

reg = o3d.pipelines.registration.registration_icp(
    pcd_drift_corrected, pcd_clean, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

print("ICP fitness 2:", reg.fitness)
print("ICP RMSE 2:", reg.inlier_rmse)
print("Estimated correction 2:\n", reg.transformation)

error = np.linalg.norm(points - drifted_points, axis=1)
print("Mean drift 2:", error.mean())
print("Max drift 2:", error.max())