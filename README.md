
# Point Cloud Drift Simulation & ICP Drift Correction

This repository contains a reproducible Python pipeline for simulating drift in Unity-generated point clouds and correcting it using Iterative Closest Point (ICP) with Open3D.

## Overview
The project implements two foundational phases:

### Phase 1 — Drift Simulation
- Loads a Unity-exported point cloud (XYZRGBT format)
- Applies a deterministic rigid drift (rotation + translation) to a selected region
- Saves both clean and drifted versions
- Visualizes the drift with Open3D

### Phase 2 — Global Drift Correction (Foundation)
- Uses point-to-plane ICP to correct the drifted cloud
- Iteratively applies correction until fitness/RMSE thresholds are satisfied
- Visualizes the corrected result

This setup creates the basis for future real-time drift detection and region-based (local) drift analysis.

## Features
- Deterministic drift using fixed random seed
- Rigid transform simulation on full cloud or masked regions
- Point-to-plane ICP for faster and more stable convergence
- Adjustable correspondence thresholds
- Per-iteration logging of ICP metrics
- Open3D visualization for clean/drifted/corrected clouds

## File Structure
```
├── test_pointcloud2.txt      # Input cloud from Unity
├── clean_output.txt          # Saved clean cloud
├── drifted_output.txt        # Saved drifted cloud
├── test2.py                  # Main script
└── README.md
```

## Requirements
- Python 3.8+
- NumPy
- Open3D

Install dependencies:
```bash
pip install numpy open3d
```

## Running the Script
```bash
python test2.py
```
This will:
1. Load the Unity point cloud
2. Simulate drift
3. Visualize clean vs. drifted clouds
4. Run ICP correction
5. Visualize the corrected alignment

## Future Work
- Streaming drift detection using timestamps
- Local region ICP for object-level drift detection
- Real-time Unity → Python integration

## License
MIT License (or specify yours)
