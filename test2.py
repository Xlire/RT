# import numpy as np
# import open3d as o3d
# import copy  # for deep copies of point clouds


# # ---------------------------------------------------------
# # 1. Load Unity point cloud
# # ---------------------------------------------------------
# data = np.loadtxt("test_pointcloud2.txt")

# points = data[:, :3]                 # X Y Z
# colors = data[:, 3:6] / 255.0        # Normalize RGB to [0,1]
# timestamps = data[:, 6]              # Time of each point


# # ---------------------------------------------------------
# # 2. Simulate object-level drift (rigid transform on a difine region)
# # ---------------------------------------------------------

# #Set random seed
# np.random.seed(0)   # or any integer you like

# # 2.1 Define a region (example: points with x > 0 and z < 1) whole plane:  np.ones(len(points), dtype=bool)
# region_mask = np.ones(len(points), dtype=bool)

# # 2.2 Choose a time threshold t0
# t0 = np.percentile(timestamps, 50)   # drift starts halfway through the scan
# time_mask = np.ones(len(points), dtype=bool)

# # Combined mask: points in region AND after t0
# drift_mask = region_mask & time_mask

# # 2.3 Generate a random rotation rigid transform (R, t)
# def random_rotation_matrix(max_angle_deg=10):
#     angle = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
#     axis = np.random.randn(3)
#     axis /= np.linalg.norm(axis)
#     ux, uy, uz = axis
#     c = np.cos(angle)
#     s = np.sin(angle)
#     R = np.array([
#         [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
#         [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),    uy*uz*(1-c) - ux*s],
#         [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)]
#     ])
#     return R

# R = random_rotation_matrix(max_angle_deg=45)  # small rotation
# direction = np.random.randn(3)
# direction /= np.linalg.norm(direction)
# t = 4 * direction  # drift magnitude ~400 cm

# #2.4 Apply rigid transform only to selected points
# drifted_points = points.copy()
# #Look into this 
# drifted_points[drift_mask] = (R @ points[drift_mask].T).T + t

# # print(drifted_points)


# # ---------------------------------------------------------
# # 3. Create clean and drifted point clouds
# # ---------------------------------------------------------
# pcd_clean = o3d.geometry.PointCloud()
# pcd_clean.points = o3d.utility.Vector3dVector(points)
# pcd_clean.colors = o3d.utility.Vector3dVector(colors)   # IMPORTANT

# pcd_drift = o3d.geometry.PointCloud()
# pcd_drift.points = o3d.utility.Vector3dVector(drifted_points)
# pcd_drift.colors = o3d.utility.Vector3dVector(colors)   # IMPORTANT

# # ---------------------------------------------------------
# # 4. Save to TXT (preserving timestamps)
# # ---------------------------------------------------------
# def save_txt_with_time(filename, pcd, timestamps):
#     pts = np.asarray(pcd.points)
#     cols = np.asarray(pcd.colors)

#     with open(filename, "w") as f:
#         for (p, c, t) in zip(pts, cols, timestamps):
#             r, g, b = (c * 255).astype(int)
#             f.write(f"{p[0]} {p[1]} {p[2]} {r} {g} {b} {t}\n")


# save_txt_with_time("clean_output.txt", pcd_clean, timestamps)
# save_txt_with_time("drifted_output.txt", pcd_drift, timestamps)

# # ---------------------------------------------------------
# # 5. Visualization (black = clean, red = drifted)
# # ---------------------------------------------------------
# pcd_clean.colors = o3d.utility.Vector3dVector(
#     np.zeros_like(points)  # black
# )

# pcd_drift.colors = o3d.utility.Vector3dVector(
#     np.tile([1, 0, 0], (points.shape[0], 1))  # red
# )

# o3d.visualization.draw_geometries([pcd_clean, pcd_drift])

# # ---------------------------------------------------------
# # 6. Phase 2 (FOUNDATION) — Repeated ICP correction
# # ---------------------------------------------------------

# # Set simple thresholds for "good alignment"
# fitness_th = 0.75    # require at least 75% good correspondences
# rmse_th    = 0.05    # require less than 5 cm average error

# # maximum correspondence distance for ICP (in meters)
# threshold = 0.1      # allow matches within 10 cm

# max_iters = 100      # safety stop (so we don’t loop forever)

# # Make a copy of the drifted cloud to correct step-by-step
# # open3d PointCloud has no clone() method, use a deep copy instead
# pcd_current = copy.deepcopy(pcd_drift)

# # Estimate normals for both clouds (required for point-to-plane ICP)
# pcd_clean.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# pcd_current.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# print("\n=== Phase 2: Iterative Correction Loop ===")

# for i in range(max_iters):
#     if i < 5:
#         threshold = 1.0
#     elif i < 100:
#         threshold = 0.8
#     else:
#         threshold = 0.5


#     # Run ICP with current estimate (point-to-plane)
#     reg = o3d.pipelines.registration.registration_icp(
#         pcd_current,
#         pcd_clean,
#         threshold,
#         np.eye(4),
#         o3d.pipelines.registration.TransformationEstimationPointToPlane()
#     )

#     fitness = reg.fitness
#     rmse = reg.inlier_rmse

#     print(f"[Iter {i}] fitness={fitness:.5f},  rmse={rmse:.6f}")

#     # Check if alignment is good enough
#     if (fitness >= fitness_th) and (rmse <= rmse_th):
#         print("✓ Alignment good — stopping correction loop.")
#         break

#     # Otherwise apply correction transform
#     T_corr = reg.transformation
#     pcd_current.transform(T_corr)
#     # Re-estimate normals after transformation
#     pcd_current.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# else:
#     print("Reached max iterations — alignment may still be imperfect.")

# # Visualize final corrected cloud
# pcd_current.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (points.shape[0], 1)))
# o3d.visualization.draw_geometries([pcd_clean, pcd_current])
# &&&&

import numpy as np
import open3d as o3d
import copy

# ---------------------------------------------------------
# Utility: build Open3D point cloud from arrays
# ---------------------------------------------------------
def make_pcd(pts, cols):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(pts)
    p.colors = o3d.utility.Vector3dVector(cols)
    return p

# ---------------------------------------------------------
# Utility: random rotation matrix
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 1. Load Unity point cloud
# ---------------------------------------------------------
data = np.loadtxt("test_pointcloud2.txt")
points = data[:, :3]
colors = data[:, 3:6] / 255.0
timestamps = data[:, 6]

np.random.seed(2)


# ---------------------------------------------------------
# 2. Phase 3-style: Blockwise drift simulation
# ---------------------------------------------------------

# Grid size (NX × NZ blocks)
NX = 2
NZ = 2

x_min, x_max = points[:,0].min(), points[:,0].max()
z_min, z_max = points[:,2].min(), points[:,2].max()

x_edges = np.linspace(x_min, x_max, NX+1)
z_edges = np.linspace(z_min, z_max, NZ+1)

drifted_points = points.copy()

block_masks = []   # store masks for later ICP correction

block_id = 0
for ix in range(NX):
    for iz in range(NZ):
        block_id += 1
        # bounds
        xb0, xb1 = x_edges[ix], x_edges[ix+1]
        zb0, zb1 = z_edges[iz], z_edges[iz+1]

        block_mask = (
            (points[:,0] >= xb0) & (points[:,0] < xb1) &
            (points[:,2] >= zb0) & (points[:,2] < zb1)
        )
        block_masks.append(block_mask)

        if block_mask.sum() == 0:
            continue

        # Random block drift
        Rb = random_rotation_matrix(max_angle_deg=25)
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        tb = 0.5 * direction   # 2 meter drift

        print(f"Block {block_id}: drift {block_mask.sum()} points")

        drifted_points[block_mask] = (Rb @ points[block_mask].T).T + tb


# ---------------------------------------------------------
# 3. Build clean & drifted clouds for visualization
# ---------------------------------------------------------
pcd_clean = make_pcd(points, colors)
pcd_drift = make_pcd(drifted_points, colors)

pcd_clean.paint_uniform_color([0,0,0])   # black
pcd_drift.paint_uniform_color([1,0,0])   # red

print("\nShowing clean (black) vs drifted (red)")
o3d.visualization.draw_geometries([pcd_clean, pcd_drift])



# ---------------------------------------------------------
# 4. Blockwise ICP correction (Phase 3 built on Phase 2 logic)
# ---------------------------------------------------------

# Phase‑2 thresholds
fitness_th = 0.75
rmse_th    = 0.05
max_iters  = 50             # per block
initial_thresholds = [5.0, 2.0, 1.0, 0.5]  # coarse → finer

corrected_points = drifted_points.copy()


print("\n=== Phase 3: Blockwise ICP Correction ===")

for block_id, mask in enumerate(block_masks, start=1):
    
    num_pts = mask.sum()
    if num_pts == 0:
        continue

    print(f"\n--- Block {block_id}: {num_pts} points ---")

    # Clean block
    clean_block = points[mask]
    color_block = colors[mask]

    # Drifted block
    drift_block = drifted_points[mask]

    # Build Open3D clouds
    pcd_clean_blk = make_pcd(clean_block, color_block)
    pcd_drift_blk = make_pcd(drift_block, color_block)

    # Normals once
    pcd_clean_blk.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
    )
    pcd_drift_blk.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
    )

    # Working copy for iterative ICP
    pcd_curr_blk = copy.deepcopy(pcd_drift_blk)

    # ---------------------------
    #   Phase‑2 ICP loop per block
    # ---------------------------
    for thr in initial_thresholds:  # coarse→fine schedule

        threshold = thr
        print(f"  Using threshold = {thr} m")

        for i in range(max_iters):

            reg = o3d.pipelines.registration.registration_icp(
                pcd_curr_blk,
                pcd_clean_blk,
                threshold,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )

            fitness = reg.fitness
            rmse    = reg.inlier_rmse

            print(f"    [Iter {i}] fitness={fitness:.4f}, rmse={rmse:.5f}")

            # Stop early if aligned
            if (fitness >= fitness_th) and (rmse <= rmse_th):
                print("      ✓ Block aligned — stopping iterations.")
                break

            # Apply correction transform
            T_corr = reg.transformation
            pcd_curr_blk.transform(T_corr)

        # end inner iter loop

    # end threshold schedule

    # Save corrected block
    corrected_points[mask] = np.asarray(pcd_curr_blk.points)

# ---------------------------------------------------------
# 5. Visualize final corrected cloud
# ---------------------------------------------------------
pcd_corrected = make_pcd(corrected_points, colors)
pcd_corrected.paint_uniform_color([0, 1, 0])  # green

print("\nShowing corrected (green) vs clean (black)")
o3d.visualization.draw_geometries([pcd_clean, pcd_corrected])

