import random
import torch
from pipelines.base_pipeline import BasePipeline
from trainers.segmentation_trainer import SegmentationTrainer
from utils.model_persistence import load_model_for_inference

class SegmentationPipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, train_dataset, validation_dataset):
        train_loader = self._get_dataloader(train_dataset, use_lesion_sampler=self.config.use_lesion_sampler)
        validation_loader = self._get_dataloader(validation_dataset)

        model = self._init_unet()
        criterion = self._get_seg_loss()
        validation_metrics = self._get_seg_validation_metrics()
        optimizer = self._get_optimizer(model.parameters())
        scheduler = self._get_scheduler(optimizer)
        logger = self._get_logger()

        trainer = SegmentationTrainer(
            config=self.config,
            model=model,
            device=self.device,
            train_loader=train_loader,
            validation_loader=validation_loader,
            data_resolution=self.data_resolution,
            criterion=criterion,
            validation_metrics=validation_metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            saving_name=self.saving_path
        )
        
        return trainer.train(epochs=self.config.epochs)
    
    def test(self, test_dataset):
        test_loader = self._get_dataloader(test_dataset)
        
        model = self._init_unet()
        validation_metrics = self._get_seg_validation_metrics()

        load_model_for_inference(model, self.saving_path)

        trainer = SegmentationTrainer(
            config=self.config,
            model=model,
            device=self.device,
            validation_metrics=validation_metrics,
            data_resolution=self.data_resolution
        )

        return trainer.test(test_loader)
    
    def predict(self, input_tensor):
        want_hr = False

        model = self._init_unet()
        load_model_for_inference(model, self.saving_path)
        model.to(self.device).eval()

        hr_image, hr_mask, lr_image, lr_mask = input_tensor

        if want_hr:
            input_image = hr_image
        else:
            input_image = lr_image

        input_image = input_image.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output_mask = model(input_image)
            predicted_mask = torch.argmax(output_mask, dim=1).squeeze(0).cpu()
            dice = self._compute_dice(predicted_mask, lr_mask)

            if want_hr:
                return {
                    'input_image': hr_image,
                    'target_mask': hr_mask,
                    'predicted_mask': predicted_mask,
                    'dice': dice
                }
            else:
                return {
                    'input_image': lr_image,
                    'target_mask': lr_mask,
                    'predicted_mask': predicted_mask,
                    'dice': dice
                }

    def predict_random(self, dataset):
        want_hr = False
        
        model = self._init_unet()
        load_model_for_inference(model, self.saving_path)
        model.to(self.device).eval()
        
        idx = random.randint(0, len(dataset) - 1)
        hr_image, hr_mask, lr_image, lr_mask = dataset[idx]

        if want_hr:
            input_image = hr_image
        else:
            input_image = lr_image

        input_image = input_image.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output_mask = model(input_image)
            predicted_mask = torch.argmax(output_mask, dim=1).squeeze(0).cpu()
            dice = self._compute_dice(predicted_mask, lr_mask)

            if want_hr:
                return {
                    'input_image': hr_image,
                    'target_mask': hr_mask,
                    'predicted_mask': predicted_mask,
                    'dice': dice
                }
            else:
                return {
                    'input_image': lr_image,
                    'target_mask': lr_mask,
                    'predicted_mask': predicted_mask,
                    'dice': dice
                }