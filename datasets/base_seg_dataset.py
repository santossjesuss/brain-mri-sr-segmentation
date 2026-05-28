from abc import ABC, abstractmethod
import os
import random
import multiprocessing
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from transforms.base_transforms import BaseTransforms

class BaseSegmentationDataset(Dataset, ABC):
    def __init__(self, is_training, scale_factor, dataset_path=None, view='axial', enforce_min_lesion_per_batch=False, batch_size=None):
        super().__init__()
        self.view = view
        self.is_training = is_training
        self.batch_size = batch_size
        self.batch_counter = multiprocessing.Value('i', 0)
        self.counter_lock = multiprocessing.Lock()
        self.enforce_min_lesion_per_batch = enforce_min_lesion_per_batch

        self.dataset_path = self._resolve_dataset_path(dataset_path=dataset_path)
        self.images_path = self._get_data_path(is_image=True)
        self.masks_path = self._get_data_path(is_image=False)
        
        self.images_names = self._get_names(data_path=self.images_path)
        self.masks_names = self._get_names(data_path=self.masks_path)
        self.images_with_lesion, self.masks_with_lesion = self._get_with_lesion()
        
        self.transforms = BaseTransforms(scale_factor=scale_factor)
        

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        if self.enforce_min_lesion_per_batch and self.batch_counter % self.batch_size == 0:
            idx = random.randint(0, len(self.images_with_lesion) - 1)
            image_path = os.path.join(self.images_path, self.images_with_lesion[idx])
            mask_path = os.path.join(self.masks_path, self.masks_with_lesion[idx])
        else:
            image_path = os.path.join(self.images_path, self.images_names[idx])
            mask_path = os.path.join(self.masks_path, self.masks_names[idx])
        
        hr_image = read_image(image_path, mode=ImageReadMode.GRAY)
        lr_image = self.transforms.downsample_image(hr_image)
        hr_mask = read_image(mask_path, mode=ImageReadMode.GRAY)
        lr_mask = self.transforms.downsample_mask(hr_mask)

        hr_image = self.transforms.normalize_image(hr_image)
        lr_image = self.transforms.normalize_image(lr_image)

        hr_mask = hr_mask.squeeze(0)
        lr_mask = lr_mask.squeeze(0)
        hr_mask = self.transforms.normalize_binary_mask(mask=hr_mask)
        lr_mask = self.transforms.normalize_binary_mask(mask=lr_mask)
        
        self._increment_counter()

        return hr_image, hr_mask, lr_image, lr_mask

    @abstractmethod
    def get_default_folder_name(self):
        pass

    def _resolve_dataset_path(self, dataset_path):
        if dataset_path is not None:
            return dataset_path
        
        base_path = os.path.dirname(__file__)
        folder_name = self.get_default_folder_name()
        
        return os.path.join(base_path, '..', 'data', folder_name)
    
    def _get_data_path(self, is_image):
        data_type = 'images' if is_image else 'labels'
        suffix = 'Tr' if self.is_training else 'Ts'
        folder_name = f'{data_type}{suffix}'

        return os.path.join(self.dataset_path, folder_name, self.view)
    
    def _get_names(self, data_path):
        return sorted(os.listdir(data_path))
        
    def _get_with_lesion(self):
        images_with_lesion = []
        masks_with_lesion = []

        for image_name, mask_name in zip(self.images_names, self.masks_names):
            mask_path = os.path.join(self.masks_path, mask_name)
            mask = read_image(mask_path, mode=ImageReadMode.GRAY)
            if mask.max() > 0:
                images_with_lesion.append(image_name)
                masks_with_lesion.append(mask_name)
        
        return images_with_lesion, masks_with_lesion
    
    def _increment_counter(self):
        with self.counter_lock:
            self.batch_counter.value += 1