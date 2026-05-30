import os
import random
import torch
from pipelines.base_pipeline import BasePipeline
from models.multi_stage_model import MultiStageModel
from trainers.multi_stage_trainer import MultiStageTrainer
from utils.model_persistence import load_model_for_inference
from transforms.base_transforms import BaseTransforms

class FrozenSRFrozenSegPipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, train_dataset, validation_dataset):
        print('This pipeline only supports testing.')
        print('It has already both models trained.')

    def test(self, test_dataset):
        test_loader = self._get_dataloader(test_dataset)

        sr_model = self._init_rcan()
        seg_model = self._init_unet()
        validation_metrics = self._get_seg_validation_metrics()

        sr_path = os.path.join(self.config.folder_name, f'{self.config.sr_name}.pth')
        seg_path = os.path.join(self.config.folder_name, f'{self.config.hr_seg_name}.pth')
        load_model_for_inference(sr_model, sr_path)
        load_model_for_inference(seg_model, seg_path)

        sr_seg_model = MultiStageModel(
            sr_model, 
            seg_model, 
            freeze_stage_1=True, 
            freeze_stage_2=True
        )

        trainer = MultiStageTrainer(
            model=sr_seg_model,
            device=self.device,
            validation_metrics=validation_metrics
        )

        return trainer.test(test_loader)
    
    def predict(self, input_data):
        pass

    def predict_random(self, dataset):
        sr_path = os.path.join(self.config.folder_name, f'{self.config.sr_name}.pth')
        seg_path = os.path.join(self.config.folder_name, f'{self.config.hr_seg_name}.pth')

        sr_model = self._init_rcan()
        seg_model = self._init_unet()
        load_model_for_inference(sr_model, sr_path)
        load_model_for_inference(seg_model, seg_path)

        sr_seg_model = MultiStageModel(
            sr_model, 
            seg_model, 
            freeze_stage_1=True, 
            freeze_stage_2=True
        ).to(self.device)
        sr_seg_model.eval()

        idx = random.randint(0, len(dataset) - 1)
        hr_image, hr_mask, lr_image, lr_mask = dataset[idx]
        transforms = BaseTransforms(self.config.scale_factor)

        input_image = lr_image
        input_image = input_image.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output_mask, _ = sr_seg_model(input_image)
            output_mask = transforms.downsample_mask(output_mask)

            return {
                'input_image': lr_image,
                'target_mask': lr_mask,
                'predicted_mask': torch.argmax(output_mask, dim=1).squeeze(0).cpu()
            }