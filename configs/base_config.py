from dataclasses import dataclass

@dataclass
class BaseConfig:
    # --|Server config|--
    # # Training config
    # epochs: int = 100
    # batch_size: int = 16
    # learning_rate: float = 1e-4
    # shuffle_data: bool = True
    # num_workers: int = 4

    # # SuperRes config
    # num_rg: int = 5
    # num_rcab: int = 8
    # sr_inner_channels: int = 64
    # --|Server config|--
 
    # --|Local config|--
    # Training config
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 1e-4
    shuffle_data: bool = True
    num_workers: int = 1

    # SuperRes config
    num_rg: int = 1
    num_rcab: int = 1
    sr_inner_channels: int = 2
    # --|Local config|--
    
    # Segmentation config
    seg_model_name: str = 'resnet34'
    seg_encoder_weights: str = None

    # Losses config
    dice_weight: float = 0.5
    cross_entropy_weight: float = 0.5
    sr_loss_weight: float = 0.5
    seg_loss_weight: float = 0.5
    tversky_alpha: float = 0.1
    tversky_beta: float = 0.9
    tversky_smooth: float = 1e-6

    # Meta-parameters config
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    max_grad_norm: float = 1.0

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