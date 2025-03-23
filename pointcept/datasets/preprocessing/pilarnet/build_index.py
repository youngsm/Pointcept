"""
Build index for PILArNet h5 files

This script counts the number of points in each event in the h5 file
and saves them to a numpy file, which is used for faster filtering
in the preprocessing pipeline.
"""

import argparse
import multiprocessing
import os
import sys

import h5py as h5
import numpy as np
from glob import glob

try:
    from tqdm import trange
except ImportError:
    def trange(*args, **kwargs):
        proc = multiprocessing.current_process()
        worker_id = proc._identity[0] - 1 if proc._identity else 0
        for i in range(args[0]):
            print(f"{worker_id} {kwargs.get('desc', '')} {i}/{args[0]}", end='\r')
            yield i

def process_file(file_path: str) -> None:
    """
    Process a PILArNet h5 file and save the number of points in each event
    for use in preprocessing.
    """
    # progress bar based on the worker's identity
    proc = multiprocessing.current_process()
    worker_id = proc._identity[0] - 1 if proc._identity else 0

    with h5.File(
        file_path,
        'r',
        libver='latest',
        swmr=True,
    ) as f:
        num_points = []
        for i in trange(
            f['cluster'].shape[0],
            desc=f'[Worker {worker_id}] {os.path.basename(file_path)}',
            ncols=80,
            position=worker_id,
            leave=False,
        ):
            cluster_size = f['cluster'][i].reshape(-1, 5)[:, 0]
            num_points.append(cluster_size.sum())

    output_path = file_path.replace(".h5", "_points.npy")
    np.save(output_path, np.array(num_points).squeeze())

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Process PILArNet event lengths for faster preprocessing.'
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help='Directory containing h5 files. Script will find all .h5 files in this directory.',
    )
    parser.add_argument(
        '-p',
        '--pattern',
        type=str,
        default="*.h5",
        help='Glob pattern to match h5 files. Default: "*.h5"',
    )
    parser.add_argument(
        '-j',
        '--num-workers',
        type=int,
        default=multiprocessing.cpu_count(),
        help='Number of workers to use for processing.',
    )
    args = parser.parse_args()

    # Find all h5 files matching the pattern
    h5_files = sorted(glob(os.path.join(args.data_dir, args.pattern)))
    
    if not h5_files:
        print(f"No files found matching pattern '{args.pattern}' in directory '{args.data_dir}'")
        sys.exit(1)
    
    print(f"Found {len(h5_files)} files to process")
    
    with multiprocessing.Pool(min(args.num_workers, len(h5_files))) as pool:
        pool.map(process_file, h5_files)

if __name__ == "__main__":
    main() 