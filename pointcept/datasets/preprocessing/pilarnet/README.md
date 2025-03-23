# PILArNet Preprocessing

This directory contains scripts for preprocessing PILArNet datasets for use with Pointcept.

## Overview

The PILArNet dataset contains point clouds from particle physics data stored in h5 files. The preprocess_pilarnet.py script converts these h5 files into the format required by Pointcept.

## Data Format

After preprocessing, the data will be structured as follows:
```
output_root/
├── train/
│   ├── filename_000000/
│   │   ├── coord.npy       # Point coordinates (N, 3)
│   │   ├── color.npy       # Log-transformed energy as RGB (N, 3)
│   │   ├── energy.npy      # Raw energy values (N, 1)
│   │   └── segment.npy     # Semantic IDs (N, 1)
│   ├── filename_000001/
│   ...
├── val/
    ...
```

## Building Index (Optional)

First, you can optionally build an index of point counts to speed up the preprocessing:

```bash
python build_index.py /path/to/pilarnet/h5/files -p "*.h5" -j 8
```

This creates a `*_points.npy` file for each h5 file, which allows the preprocessing script to quickly filter point clouds with too few points.

### Parameters

- `data_dir`: Directory containing h5 files
- `-p, --pattern`: Glob pattern to match h5 files (default: "*.h5")
- `-j, --num-workers`: Number of workers for parallel processing

## Preprocessing

To preprocess PILArNet data:

```bash
python preprocess_pilarnet.py \
    --dataset_root /path/to/pilarnet/h5/files \
    --output_root /path/to/output/directory \
    --train_pattern "*train*.h5" \
    --val_pattern "*val*.h5" \
    --min_points 1024 \
    --energy_threshold 0.0 \
    --num_workers 8
```

### Parameters

- `--dataset_root`: Path to the directory containing PILArNet h5 files
- `--output_root`: Output directory where the processed data will be saved
- `--train_pattern`: Glob pattern to identify training h5 files (default: "*train*.h5")
- `--val_pattern`: Glob pattern to identify validation h5 files (default: "*val*.h5")
- `--min_points`: Minimum number of points required for a point cloud (default: 1024)
- `--energy_threshold`: Energy threshold for filtering points (0.0 to keep all points)
- `--num_workers`: Number of worker processes for parallel preprocessing

## Semantic Labels

The PILArNet dataset includes these semantic classes:
- 0: Shower
- 1: Track
- 2: Michel
- 3: Delta
- 4: Low energy deposit 