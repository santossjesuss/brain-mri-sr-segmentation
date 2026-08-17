import cv2
import torch
import torch.nn.functional as F
import albumentations as A

class BaseTransforms():
    def __init__(self, scale_factor, target_width=None, target_height=None):
        self.scale_factor = scale_factor
        self.downsample_factor = self._calc_downsample_factor(scale_factor)
        self.target_dimensions = (target_width, target_height)
        self.img_downsample_mode_torch = 'bicubic'
        self.mask_downsample_mode_torch = 'nearest'
        self.img_downsample_mode_cv = cv2.INTER_AREA
        self.mask_downsample_mode_cv = cv2.INTER_NEAREST
        self.probability_horizontal_flip = 0.5
        self.train_augment = A.HorizontalFlip(p=self.probability_horizontal_flip)

    def normalize_image(self, image):
        return image.float() / 255.0

    def normalize_binary_mask(self, mask):
        return (mask > 0).long()

    def downsample_image_torch(self, image):
        image = image.unsqueeze(0)
        lr_image = F.interpolate(image, scale_factor=self.downsample_factor, mode=self.img_downsample_mode_torch)

        return lr_image.squeeze(0)

    def downsample_mask_torch(self, mask):
        mask = mask.unsqueeze(0)
        lr_mask = F.interpolate(mask, scale_factor=self.downsample_factor, mode=self.mask_downsample_mode_torch)

        return lr_mask.squeeze(0)

    def downsample_image_cv(self, image):
        image = image.squeeze(0).cpu().numpy()
        lr_image = cv2.resize(image, self.target_dimensions, interpolation=self.img_downsample_mode_cv)

        return torch.from_numpy(lr_image).unsqueeze(0)

    def downsample_mask_cv(self, mask):
        mask = mask.squeeze(0).cpu().numpy()
        lr_mask = cv2.resize(mask, self.target_dimensions, interpolation=self.mask_downsample_mode_cv)

        return torch.from_numpy(lr_mask).unsqueeze(0)

    def augment(self, image, mask):
        augmented = self.train_augment(image=image, mask=mask)

        return augmented['image'], augmented['mask']
    
    def _calc_downsample_factor(self, scale_factor):
        return 1 / scale_factor