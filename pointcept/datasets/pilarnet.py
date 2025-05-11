"""
PILArNet Dataset

This module handles the PILArNet dataset for particle physics point cloud segmentation.
"""

import os
import glob
import numpy as np
import torch
import h5py
from copy import deepcopy
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import Compose, TRANSFORMS

@DATASETS.register_module()
class PILArNetH5Dataset(Dataset):
    """
    PILArNet Dataset that loads directly from h5 files, avoiding the need for preprocessing to individual files.
    
    The dataset contains the following semantic classes:
    - 0: Shower
    - 1: Track
    - 2: Michel
    - 3: Delta
    - 4: Low energy deposit
    """
    
    # Map of PILArNet semantic classes to IDs
    class2id = np.array([0, 1, 2, 3, 4])
    
    def __init__(
        self,
        data_root,
        split="train",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
        energy_threshold=0.0,
        min_points=1024,
        max_len=-1,
        remove_low_energy_scatters=False,
        generate_queries=False,
        num_queries=5,
        query_mask_ratio=0.3,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.loop = loop if not test_mode else 1
        self.ignore_index = ignore_index

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        # PILArNet specific parameters
        self.energy_threshold = energy_threshold
        self.min_points = min_points
        self.remove_low_energy_scatters = remove_low_energy_scatters
        self.max_len = max_len
        # Get list of h5 files
        self.h5_files = self.get_h5_files()
        assert len(self.h5_files) > 0, "No h5 files found"
        self.initted = False
        
        # Build index for faster access
        self._build_index()
        
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in PILArNet {} set.".format(
                self.cumulative_lengths[-1], self.loop, split
            )
        )
        
        # Instance segmentation parameters
        self.generate_queries = generate_queries
        self.num_queries = num_queries
        self.query_mask_ratio = query_mask_ratio
    
    def get_h5_files(self):
        """Get list of h5 files based on the split."""
        if isinstance(self.split, str):
            split_pattern = f"*{self.split}/*.h5"
        else:
            split_pattern = [f"*{s}/*.h5" for s in self.split]
            
        if isinstance(split_pattern, list):
            h5_files = []
            for pattern in split_pattern:
                h5_files.extend(glob.glob(os.path.join(self.data_root, pattern)))
        else:
            h5_files = glob.glob(os.path.join(self.data_root, split_pattern))
            
        return sorted(h5_files)
    
    def _build_index(self):
        """Build an index of valid point clouds for faster access."""
        log = get_root_logger()
        log.info("Building index for PILArNetH5Dataset")
        
        self.cumulative_lengths = []
        self.indices = []
        
        for h5_file in self.h5_files:
            try:
                # Check if points count file exists
                points_file = h5_file.replace(".h5", "_points.npy")
                if os.path.exists(points_file):
                    npoints = np.load(points_file)
                    index = np.argwhere(npoints >= self.min_points).flatten()
                else:
                    # No points file, count on the fly
                    log.info(f"No points count file for {h5_file}, counting points on the fly")
                    with h5py.File(h5_file, 'r', libver='latest', swmr=True) as f:
                        # Get all point counts
                        npoints = []
                        for i in range(f['cluster'].shape[0]):
                            cluster_size = f['cluster'][i].reshape(-1, 5)[:, 0]
                            npoints.append(cluster_size.sum())
                        npoints = np.array(npoints)
                        index = np.argwhere(npoints >= self.min_points).flatten()
            except Exception as e:
                log.warning(f"Error processing {h5_file}: {e}")
                index = np.array([])
                
            self.cumulative_lengths.append(index.shape[0])
            self.indices.append(index)
            
        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        log.info(f"Found {self.cumulative_lengths[-1]} point clouds with at least {self.min_points} points")
    
    def h5py_worker_init(self):
        """Initialize h5py files for each worker."""
        self.h5data = []
        for h5_file in self.h5_files:
            self.h5data.append(h5py.File(h5_file, mode="r", libver="latest", swmr=True))
        self.initted = True
    
    def get_queries(self, data_dict):
        """Generate instance segmentation queries from a point cloud.
        
        Args:
            data_dict (dict): Dictionary containing point cloud data including instances
            
        Returns:
            list: List of data dictionaries, one for each query
        """
        if not self.generate_queries:
            return [data_dict]

        instance_ids = np.unique(data_dict["instance"].flatten())
        instance_ids = instance_ids[instance_ids > 0]  # Exclude background/invalid
        if len(instance_ids) > self.num_queries:
            selected_ids = np.random.choice(instance_ids, self.num_queries, replace=False)
        else:
            selected_ids = instance_ids[:self.num_queries]

        result = []
        for instance_id in selected_ids:
            instance_dict = deepcopy(data_dict)            
            full_mask = (instance_dict["instance"].flatten() == instance_id).astype(np.float32)
            
            # create corrupted query mask
            mask_indices = np.where(full_mask)[0]
            if len(mask_indices) > 0:
                keep_indices = np.random.choice(
                    mask_indices, 
                    size=max(1, int(self.query_mask_ratio * len(mask_indices))), 
                    replace=False
                )
                query_mask = np.zeros_like(full_mask)
                query_mask[keep_indices] = 1.0
                instance_dict["query_mask"] = query_mask[:, None] # (N, 1)
                instance_dict["query_truth"] = full_mask[:, None] # (N, 1)
                instance_dict["query_instance_id"] = np.array([instance_id])
                result.append(instance_dict)
        
        if not result:
            data_dict["query_mask"] = np.zeros((data_dict["instance"].shape[0], 1), dtype=np.float32)
            data_dict["query_truth"] = np.zeros((data_dict["instance"].shape[0], 1), dtype=np.float32)
            data_dict["query_instance_id"] = np.array([-1])
            result = [data_dict]
            
        return result
    
    def get_data(self, idx):
        """Load a point cloud from h5 file."""
        if not self.initted:
            self.h5py_worker_init()
            
        # Find which h5 file and index the point cloud is in
        h5_idx = np.searchsorted(self.cumulative_lengths, idx, side="right")
        if h5_idx > 0:
            idx_in_file = idx - self.cumulative_lengths[h5_idx - 1]
        else:
            idx_in_file = idx
            
        h5_file = self.h5data[h5_idx]
        file_idx = self.indices[h5_idx][idx_in_file]
        
        # Load point cloud data
        data = h5_file["point"][file_idx].reshape(-1, 8)[:, [0, 1, 2, 3]] # (x,y,z,e)
        cluster_size, group_id, semantic_id = h5_file["cluster"][file_idx].reshape(-1, 5)[:, [0, 2, -1]].T
        
        # Remove low energy scatters if configured
        if self.remove_low_energy_scatters:
            data = data[cluster_size[0]:]
            semantic_id, group_id, cluster_size = semantic_id[1:], group_id[1:], cluster_size[1:]
        
        # Compute semantic ids for each point
        data_semantic_id = np.repeat(semantic_id, cluster_size)
        data_group_id = np.repeat(group_id, cluster_size)

        # Apply energy threshold if needed
        if self.energy_threshold > 0:
            threshold_mask = data[:, 3] > self.energy_threshold
            data = data[threshold_mask]
            data_semantic_id = data_semantic_id[threshold_mask]
            data_group_id = data_group_id[threshold_mask]
            
            if data.shape[0] < self.min_points:
                # Try another data point if this one is too small after filtering
                return self.get_data((idx + 1) % len(self))
        
        # Prepare return dictionary
        data_dict = {}
        
        # Get coordinates
        data_dict["coord"] = data[:, :3].astype(np.float32)
        
        # Process energy (raw and normalized)
        energy = data[:, 3].astype(np.float32)
        data_dict["energy"] = energy[:, None]
        
        # Get semantic labels
        data_dict["segment"] = data_semantic_id.astype(np.int32)[:, None]
        
        # Dummy instance for compatibility
        data_dict["instance"] = data_group_id.astype(np.int32)[:, None]
        
        # Add metadata
        h5_name = os.path.basename(self.h5_files[h5_idx])
        data_dict["name"] = f"{h5_name}_{file_idx}"
        data_dict["split"] = self.split if isinstance(self.split, str) else "custom"
        
        return data_dict
    
    def get_data_name(self, idx):
        """Get name for the point cloud."""
        if not self.initted:
            self.h5py_worker_init()
            
        # Find which h5 file and index the point cloud is in
        h5_idx = np.searchsorted(self.cumulative_lengths, idx, side="right")
        if h5_idx > 0:
            idx_in_file = idx - self.cumulative_lengths[h5_idx - 1]
        else:
            idx_in_file = idx
            
        h5_name = os.path.basename(self.h5_files[h5_idx])
        file_idx = self.indices[h5_idx][idx_in_file]
        
        return f"{h5_name}_{file_idx}"
    
    def prepare_train_data(self, idx):
        """Prepare training data with transforms."""
        data_dict = self.get_data(idx % len(self))
        if self.generate_queries:
            data_dicts = self.get_queries(data_dict)
            if self.transform is not None:
                data_dicts = [self.transform(data_dict) for data_dict in data_dicts]
            return data_dicts
        return self.transform(data_dict)
                
    def prepare_test_data(self, idx):
        """Prepare test data with test transforms."""
        # Load data
        data_dict = self.get_data(idx % len(self))
        
        # Apply transforms
        if self.transform is not None:
            data_dict = self.transform(data_dict)
            
        # Test mode specific handling
        result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = self.get_queries(fragment_list)
        return result_dict
        
    def __getitem__(self, idx):
        real_idx = idx % len(self)
        if self.test_mode:
            return self.prepare_test_data(real_idx)
        else:
            return self.prepare_train_data(real_idx)
    
    def __len__(self):
        if self.max_len > 0:
            return min(self.max_len, self.cumulative_lengths[-1]) * self.loop
        return self.cumulative_lengths[-1] * self.loop
    
    def __del__(self):
        """Clean up open h5 files."""
        if hasattr(self, 'initted') and self.initted:
            for h5_file in self.h5data:
                h5_file.close()
    
    @staticmethod
    def worker_init_fn(worker_id):
        """Initialize worker with unique seed and open h5 files."""
        np.random.seed(np.random.get_state()[1][0] + worker_id)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            dataset = worker_info.dataset
            if isinstance(dataset, PILArNetH5Dataset):
                dataset.h5py_worker_init()
            else:
                # Handle case where the dataset is wrapped in a ConcatDataset
                for d in getattr(dataset, 'datasets', []):
                    if isinstance(d, PILArNetH5Dataset):
                        d.h5py_worker_init() 