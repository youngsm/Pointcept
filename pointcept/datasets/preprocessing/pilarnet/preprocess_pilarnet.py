"""
Preprocessing Script for PILArNet Dataset

This script converts PILArNet h5 files to the Pointcept format.
"""

import os
import argparse
import glob
import h5py
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

def log_transform(x, xmax=20.0, eps=1e-2):
    """Transform energy to logarithmic scale on [-1,1]"""
    # [eps, xmax] -> [-1,1]
    y0 = np.log10(eps)
    y1 = np.log10(eps + xmax)
    return 2 * (np.log10(x + eps) - y0) / (y1 - y0) - 1

def process_file(file_path, output_root, split_name, min_points=1024, energy_threshold=0.0):
    """Process a PILArNet h5 file and save the data in Pointcept format."""
    print(f"Processing: {os.path.basename(file_path)} in {split_name}")
    
    try:
        with h5py.File(
            file_path,
            'r',
            libver='latest',
            swmr=True,
        ) as f:
            # Get number of point clouds in the file
            num_clouds = f['point'].shape[0]
            
            # Create output directory for this file
            base_name = os.path.basename(file_path).replace(".h5", "")
            output_dir = os.path.join(output_root, split_name)
            os.makedirs(output_dir, exist_ok=True)
            
            for i in range(num_clouds):
                # Get point cloud data
                data = f['point'][i].reshape(-1, 8)[:, [0, 1, 2, 3, 5]]  # (x,y,z,e,t)
                cluster_size, semantic_id = f['cluster'][i].reshape(-1, 5)[:, [0, -1]].T
                
                # Generate semantic IDs for each point
                data_semantic_id = np.repeat(semantic_id, cluster_size)
                
                # Save original coords
                coords = data[:, :3].astype(np.float32)
                energies = data[:, 3]
                
                # Filter out point clouds with too few points
                if data.shape[0] < min_points:
                    continue
                
                # Create point cloud directory
                cloud_dir = os.path.join(output_dir, f"{base_name}_{i:06d}")
                os.makedirs(cloud_dir, exist_ok=True)
                
                # Apply energy threshold if needed
                if energy_threshold > 0.0:
                    threshold_mask = energies > energy_threshold
                    if np.sum(threshold_mask) < min_points:
                        continue  # Skip if too few points after filtering
                    
                    coords = coords[threshold_mask]
                    energies = energies[threshold_mask]
                    data_semantic_id = data_semantic_id[threshold_mask]
                
                # Save coordinates
                np.save(os.path.join(cloud_dir, "coord.npy"), coords)
                
                # Convert energy to RGB-like format (using log-transformed energy in all 3 channels)
                log_energy = log_transform(energies)
                # Scale to 0-255 range for RGB
                log_energy_color = ((log_energy + 1) / 2 * 255).astype(np.uint8)
                colors = np.column_stack([log_energy_color, log_energy_color, log_energy_color])
                np.save(os.path.join(cloud_dir, "color.npy"), colors)
                
                # Save raw energy values as a separate feature
                np.save(
                    os.path.join(cloud_dir, "energy.npy"), log_energy.astype(np.float32)
                )
                
                # Save semantic IDs
                np.save(os.path.join(cloud_dir, "segment.npy"), data_semantic_id.astype(np.int16))
                
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the PILArNet dataset containing h5 files",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located",
    )
    parser.add_argument(
        "--train_pattern",
        default="*train*.h5",
        help="Pattern to identify training h5 files",
    )
    parser.add_argument(
        "--val_pattern",
        default="*val*.h5",
        help="Pattern to identify validation h5 files",
    )
    parser.add_argument(
        "--min_points",
        default=1024,
        type=int,
        help="Minimum number of points for a point cloud",
    )
    parser.add_argument(
        "--energy_threshold",
        default=0.0,
        type=float,
        help="Energy threshold for points (0.0 to keep all points)",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    config = parser.parse_args()

    # Create output directories
    train_output_dir = os.path.join(config.output_root, "train")
    os.makedirs(train_output_dir, exist_ok=True)
    val_output_dir = os.path.join(config.output_root, "val")
    os.makedirs(val_output_dir, exist_ok=True)

    # Find h5 files
    train_files = sorted(glob.glob(os.path.join(config.dataset_root, config.train_pattern)))
    val_files = sorted(glob.glob(os.path.join(config.dataset_root, config.val_pattern)))
    
    print(f"Found {len(train_files)} training files and {len(val_files)} validation files")
    
    # Preprocess data
    print("Processing files...")
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    
    # Process training files
    _ = list(
        pool.map(
            process_file,
            train_files,
            repeat(config.output_root),
            repeat("train"),
            repeat(config.min_points),
            repeat(config.energy_threshold),
        )
    )
    
    # Process validation files
    _ = list(
        pool.map(
            process_file,
            val_files,
            repeat(config.output_root),
            repeat("val"),
            repeat(config.min_points),
            repeat(config.energy_threshold),
        )
    )

if __name__ == "__main__":
    main() 