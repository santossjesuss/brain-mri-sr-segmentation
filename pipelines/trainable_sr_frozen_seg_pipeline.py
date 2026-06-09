import os
import random
import torch
from pipelines.base_pipeline import BasePipeline
from models.multi_stage_model import MultiStageModel
from trainers.multi_stage_trainer import MultiStageTrainer
from utils.model_persistence import load_model_for_inference
from transforms.base_transforms import BaseTransforms

class TrainableSRFrozenSegPipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, train_dataset, validation_dataset):
        train_loader = self._get_dataloader(train_dataset, use_lesion_sampler=self.config.use_lesion_sampler)
        validation_loader = self._get_dataloader(validation_dataset)

        sr_model = self._init_rcan()
        seg_model = self._init_unet()
        criterion = self._get_seg_loss()
        validation_metrics = self._get_seg_validation_metrics()
        logger = self._get_logger()
        img_logger = self._get_img_logger()

        seg_path = os.path.join(self.config.folder_name, f'{self.config.hr_seg_name}.pth')
        load_model_for_inference(model=seg_model, saving_name=seg_path)

        trainable_sr_frozen_seg_model = MultiStageModel(
            sr_model, 
            seg_model, 
            freeze_stage_1=False, 
            freeze_stage_2=True
        )
        optimizer = self._get_optimizer(trainable_sr_frozen_seg_model.parameters())
        scheduler = self._get_scheduler(optimizer)

        trainer = MultiStageTrainer(
            config=self.config,
            model=trainable_sr_frozen_seg_model,
            device=self.device,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            validation_metrics=validation_metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            img_logger=img_logger,
            saving_name=self.saving_path
        )

        return trainer.train(epochs=self.config.epochs)
    
    def test(self, test_dataset):
        test_loader = self._get_dataloader(test_dataset)

        sr_model = self._init_rcan()
        seg_model = self._init_unet()
        validation_metrics = self._get_seg_validation_metrics()

        trainable_sr_frozen_seg_model = MultiStageModel(
            sr_model, 
            seg_model, 
            freeze_stage_1=True, 
            freeze_stage_2=True
        )

        load_model_for_inference(model=trainable_sr_frozen_seg_model, saving_name=self.saving_path)

        trainer = MultiStageTrainer(
            config=self.config,
            model=trainable_sr_frozen_seg_model,
            device=self.device,
            validation_metrics=validation_metrics
        )

        return trainer.test(test_loader)
    
    def predict(self, input_tensor):
        sr_model = self._init_rcan()
        seg_model = self._init_unet()

        trainable_sr_frozen_seg_model = MultiStageModel(
            sr_model, 
            seg_model, 
            freeze_stage_1=True, 
            freeze_stage_2=True
        )
        load_model_for_inference(model=trainable_sr_frozen_seg_model, saving_name=self.saving_path)
        trainable_sr_frozen_seg_model.to(self.device).eval()

        hr_image, hr_mask, lr_image, lr_mask = input_tensor
        transforms = BaseTransforms(self.config.scale_factor)

        input_image = lr_image
        input_image = input_image.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output_mask, _ = trainable_sr_frozen_seg_model(input_image)
            output_mask = torch.argmax(output_mask, dim=1).float()
            output_mask = transforms.downsample_mask(output_mask)
            predicted_mask = output_mask.long().squeeze(0).cpu()
            dice = self._compute_dice(predicted_mask, lr_mask)

            return {
                'input_image': lr_image,
                'target_mask': lr_mask,
                'predicted_mask': predicted_mask,
                'dice': dice
            }

    def predict_random(self, dataset):
        sr_model = self._init_rcan()
        seg_model = self._init_unet()

        trainable_sr_frozen_seg_model = MultiStageModel(
            sr_model, 
            seg_model, 
            freeze_stage_1=True, 
            freeze_stage_2=True
        )
        load_model_for_inference(model=trainable_sr_frozen_seg_model, saving_name=self.saving_path)
        trainable_sr_frozen_seg_model.to(self.device).eval()

        idx = random.randint(0, len(dataset) - 1)
        hr_image, hr_mask, lr_image, lr_mask = dataset[idx]
        transforms = BaseTransforms(self.config.scale_factor)

        input_image = lr_image
        input_image = input_image.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output_mask, _ = trainable_sr_frozen_seg_model(input_image)
            output_mask = torch.argmax(output_mask, dim=1).float()
            output_mask = transforms.downsample_mask(output_mask)
            predicted_mask = output_mask.long().squeeze(0).cpu()
            dice = self._compute_dice(predicted_mask, lr_mask)

            return {
                'input_image': lr_image,
                'target_mask': lr_mask,
                'predicted_mask': predicted_mask,
                'dice': dice
            }