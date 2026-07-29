from dataclasses import dataclass

@dataclass
class BaseConfig:
    # --|Server config|--
    # Training config
    epochs: int = 100
    batch_size: int = 16
    num_workers: int = 4
    shuffle_data: bool = True

    # SuperRes config
    num_rg: int = 5
    num_rcab: int = 8
    sr_inner_channels: int = 64
    # --|Server config|--
 
    # --|Local config|--
    # Training config
    # epochs: int = 1
    # batch_size: int = 2
    # num_workers: int = 1
    # shuffle_data: bool = True

    # # SuperRes config
    # num_rg: int = 1
    # num_rcab: int = 1
    # sr_inner_channels: int = 2
    # --|Local config|--
    
    # Segmentation config
    seg_model_name: str = 'resnet34'
    # seg_encoder_weights: str = None       # pre-cambios de Karl 
    seg_encoder_weights: str = 'imagenet'   # cambio sugerido

    # Losses config
    sr_loss_weight: float = 0.5
    seg_loss_weight: float = 0.5

    # Meta-parameters config
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    max_grad_norm: float = 1.0

    # Hiperparameter Tuning
    # --|V.1 Hiperparameters|--
    learning_rate: float = 1e-4
    dice_weight: float = 0.5
    cross_entropy_weight: float = 0.5

    # --|V.2 Hiperparameters|--
    # learning_rate: float = 3e-4
    # dice_weight: float = 0.5
    # cross_entropy_weight: float = 0.5

    # --|V.3 Hiperparameters|--
    # learning_rate: float = 3e-4
    # dice_weight: float = 0.7
    # cross_entropy_weight: float = 0.3


    # Saving config
    folder_name: str = 'trained_models'
    sr_name: str = 'sr'
    hr_seg_name: str = 'hr_seg'
    lr_seg_name: str = 'lr_seg'
    frozen_sr_frozen_seg: str = 'frozen_sr_frozen_seg'
    frozen_sr_trainable_seg: str = 'frozen_sr_trainable_seg'
    trainable_sr_frozen_seg: str = 'trainable_sr_frozen_seg'
    joint_sr_seg_e2e: str = 'joint_sr_seg_e2e'
    joint_sr_seg_combined: str = 'joint_sr_seg_combined'

    # Logging config
    base_img_log_dir: str = 'logs/images'