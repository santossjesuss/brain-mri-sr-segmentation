import os
import numpy as np
from torch.utils.data import BatchSampler
from torchvision.io import read_image, ImageReadMode

class LesionGuaranteedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, positives_per_batch=1, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.positives_per_batch = positives_per_batch
        self.drop_last = drop_last

        self.dataset_indices, self.positive_indices = self._scan_training_subset()

    def __iter__(self):
        num_batches = self.__len__()
        total_indices = len(self.dataset_indices)
        dataset_indices = self._shuffle_indices(self.dataset_indices)
        positive_indices = self._shuffle_indices(self.positive_indices)
        random_slots = self.batch_size - self.positives_per_batch
        pos_idx = 0
        dataset_idx = 0

        for _ in range(num_batches):
            batch = []

            for _ in range(self.positives_per_batch):
                choose_idx = pos_idx % len(positive_indices)
                batch.append(positive_indices[choose_idx])
                pos_idx += 1

            for _ in range(random_slots):
                choose_idx = dataset_idx % total_indices
                batch.append(dataset_indices[choose_idx])
                dataset_idx += 1

            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        total_samples = len(self.dataset)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return -(-total_samples // self.batch_size)

    def _scan_dataset(self):
        dataset = self._get_correct_dataset()
        masks_path = dataset.mask_paths
        mask_names = dataset.mask_names
        
        dataset_indices = []
        positive_indices = []
        for idx, name in enumerate(mask_names):
            dataset_indices.append(idx)
            path = os.path.join(masks_path, name)
            mask = read_image(path, mode=ImageReadMode.GRAY)
            if mask.any():
                print(f"Sample mask stats - dtype: {mask.dtype}, min: {mask.min()}, max: {mask.max()}, unique: {mask.unique()}")
                positive_indices.append(idx)

        print(f"Found {len(positive_indices)} positives out of {len(mask_names)}")
        return dataset_indices, positive_indices
    
    def _scan_training_subset(self):
        dataset = self._get_correct_dataset()
        training_subset_indices = self.dataset.indices
        masks_path = dataset.mask_paths
        mask_names = dataset.mask_names
        
        dataset_indices = []
        positive_indices = []
        for idx in training_subset_indices:
            dataset_indices.append(idx)
            name = mask_names[idx]
            path = os.path.join(masks_path, name)
            mask = read_image(path, mode=ImageReadMode.GRAY)
            if mask.any():
                print(f"Sample mask stats - dtype: {mask.dtype}, min: {mask.min()}, max: {mask.max()}, unique: {mask.unique()}")
                positive_indices.append(idx)

        print(f"Found {len(positive_indices)} positives out of {len(mask_names)}")
        return dataset_indices, positive_indices
    
    def _get_correct_dataset(self):
        if hasattr(self.dataset, 'dataset'):
            return self.dataset.dataset
        
        return self.dataset
    
    @staticmethod
    def _shuffle_indices(indices):
        arr = np.array(indices)
        np.random.shuffle(arr)
        return arr.tolist()