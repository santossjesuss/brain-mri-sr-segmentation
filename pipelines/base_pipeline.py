from abc import ABC, abstractmethod
import os
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from configs.mslesseg_config import MSLesSegConfig
from torch.utils.data import DataLoader
from samplers.lesion_guaranteed_sampler import LesionGuaranteedBatchSampler
import segmentation_models_pytorch as smp
from models.rcan.rcan import RCAN
from losses.dice_ce_combined_loss import DiceCECombinedLoss
from losses.sr_seg_combined_loss import SRSegCombinedLoss
from metrics.superres_metrics import SuperResolutionMetrics
from metrics.segmentation_metrics import SegmentationMetrics
from loggers.tensorboard_logger import TensorBoardLogger
from loggers.image_logger import ImageLogger
from utils.gpu import enable_cuda

class BasePipeline(ABC):
    def __init__(self, config, experiment_name, dataset_name=None, data_resolution=None):
        self.config = config
        self.device = enable_cuda()
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.data_resolution = data_resolution
        self.saving_path = os.path.join(self.config.folder_name, f'{experiment_name}.pth')

    @abstractmethod
    def run(self, train_dataset, validation_dataset):
        pass

    @abstractmethod
    def test(self, test_dataset):
        pass

    @abstractmethod
    def predict(self, input_tensor):
        pass

    @abstractmethod
    def predict_random(self, dataset):
        pass

    def _get_dataloader(self, dataset, use_lesion_sampler=False):
        if use_lesion_sampler:
            sampler = LesionGuaranteedBatchSampler(
                dataset=dataset,
                batch_size=self.config.batch_size,
                positives_per_batch=self.config.positives_per_batch,
                drop_last=self.config.drop_last
            )
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=self.config.num_workers
            )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_data,
            num_workers=self.config.num_workers
        )

    def _init_rcan(self):
        return RCAN(
            in_channels=self.config.in_channels,
            out_channels=self.config.sr_out_channels,
            num_rg=self.config.num_rg,
            num_rcab=self.config.num_rcab,
            channels=self.config.sr_inner_channels,
            upscale_factor=self.config.scale_factor
        )
    
    def _init_unet(self):
        return smp.Unet(
            encoder_name=self.config.seg_model_name,
            encoder_weights=self.config.seg_encoder_weights,
            in_channels=self.config.in_channels,
            classes=self.config.seg_classes
        )
    
    def _get_sr_loss(self):
        return nn.L1Loss()

    def _get_seg_loss(self):
        if self.dataset_name == MSLesSegConfig.dataset_name:
            return self._get_dice_ce_combined_loss()
        elif self.dataset_name == FCDLesSegConfig.dataset_name:
            return self._get_focal_tversky_loss()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def _get_dice_ce_combined_loss(self):
        return DiceCECombinedLoss(
            dice_weight=self.config.dice_weight, 
            cross_entropy_weight=self.config.cross_entropy_weight
        )
        
    def _get_focal_tversky_loss(self):
        return smp.losses.TverskyLoss(
            mode='multiclass', #might change to 'binary'
            alpha=self.config.tversky_alpha,
            beta=self.config.tversky_beta,
            smooth=self.config.tversky_smooth
        )

    def _get_combined_sr_seg_loss(self):
        sr_loss = self._get_sr_loss()
        seg_loss = self._get_seg_loss()
        
        return SRSegCombinedLoss(
            sr_loss_fn=sr_loss,
            seg_loss_fn=seg_loss,
            sr_weight=self.config.sr_loss_weight,
            seg_weight=self.config.seg_loss_weight
        )

    def _get_sr_validation_metrics(self):
        return SuperResolutionMetrics()

    def _get_seg_validation_metrics(self):
        return SegmentationMetrics(
            num_classes=self.config.seg_classes
        )

    def _get_optimizer(self, model_params):
        return optim.Adam(
            model_params,
            lr=self.config.learning_rate
        )
    
    def _get_scheduler(self, optimizer):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience
        )
    
    def _get_gradient_clipper(self, model):
        return lambda: clip_grad_norm_(model.parameters(), max_norm=self.config.max_grad_norm)
    
    def _get_logger(self):
        log_path = os.path.join("logs", self.experiment_name)
        return TensorBoardLogger(log_dir=log_path)
    
    def _get_img_logger(self):
        return ImageLogger(
            base_img_log_dir=self.config.base_img_log_dir,
            model_name=self.experiment_name
        )
    
    def _compute_dice(self, predicted_mask, true_mask):
        metrics = self._get_seg_validation_metrics()
        metrics.update(
            predicted_mask.unsqueeze(0),
            true_mask.unsqueeze(0).long()
        )

        return metrics.compute()['dice']