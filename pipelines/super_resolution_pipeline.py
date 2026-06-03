import random
import torch
from pipelines.base_pipeline import BasePipeline
from trainers.super_resolution_trainer import SuperResolutionTrainer
from utils.model_persistence import load_model_for_inference

class SuperResolutionPipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, train_dataset, validation_dataset):
        train_loader = self._get_dataloader(train_dataset)
        validation_loader = self._get_dataloader(validation_dataset)

        model = self._init_rcan()
        criterion = self._get_sr_loss()
        validation_metrics = self._get_sr_validation_metrics()
        optimizer = self._get_optimizer(model.parameters())
        scheduler = self._get_scheduler(optimizer)
        logger = self._get_logger()

        trainer = SuperResolutionTrainer(
            config=self.config,
            model=model,
            device=self.device,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            validation_metrics=validation_metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            saving_name=self.saving_path,
            logger=logger
        )

        return trainer.train(epochs=self.config.epochs)
    
    def test(self, test_dataset):
        test_loader = self._get_dataloader(test_dataset)

        model = self._init_rcan()
        validation_metrics = self._get_sr_validation_metrics()

        load_model_for_inference(model, self.saving_path)

        trainer = SuperResolutionTrainer(
            config=self.config,
            model=model,
            device=self.device,
            validation_metrics=validation_metrics
        )

        return trainer.test(test_loader)
    
    def predict(self, input_tensor):
        model = self._init_rcan()
        load_model_for_inference(model, self.saving_path)
        model.to(self.device).eval()

        hr_image, hr_mask, lr_image, lr_mask = input_tensor

        input_image = lr_image
        input_image = input_image.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output_image = model(input_image)

            return {
                'input_image': lr_image,
                'target_image': hr_image,
                'predicted_image': output_image.squeeze(0).cpu()
            }

    def predict_random(self, dataset):
        model = self._init_rcan()
        load_model_for_inference(model, self.saving_path)
        model.to(self.device).eval()

        idx = random.randint(0, len(dataset) - 1)
        hr_image, hr_mask, lr_image, lr_mask = dataset[idx]

        input_image = lr_image
        input_image = input_image.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output_image = model(input_image)

            return {
                'input_image': lr_image,
                'target_image': hr_image,
                'predicted_image': output_image.squeeze(0).cpu()
            }