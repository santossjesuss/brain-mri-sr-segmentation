import torch
from tqdm import tqdm
from trainers.base_trainer import BaseTrainer
from transforms.base_transforms import BaseTransforms

class MultiStageTrainer(BaseTrainer):
    def __init__(self, *args, use_combined_loss=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_combined_loss = use_combined_loss
        self.transform = BaseTransforms(scale_factor=self.config.scale_factor)

    def _train_epoch(self, epoch, total_epochs):
        self.model.train()
        epoch_loss = 0

        progress_bar_description = f"Epoch {epoch+1}/{total_epochs}"
        progress_bar = tqdm(self.train_loader, desc=progress_bar_description)
        for batch_idx, batch in enumerate(progress_bar):
            lr_image, _, hr_image, hr_masks = self._prepare_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)
            pred_hr_masks_logits, pred_hr_image = self.model(lr_image)
            
            self.img_logger.log_batch_images(pred_hr_image, epoch, batch_idx)
            
            if self.use_combined_loss:
                loss = self.criterion(pred_hr_image, hr_image, pred_hr_masks_logits, hr_masks)
            else:
                loss = self.criterion(pred_hr_masks_logits, hr_masks)
            
            loss.backward()
            if self.gradient_clipper:
                self.gradient_clipper()
            self.optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return epoch_loss / len(self.train_loader)

    def _evaluate(self, dataloader, description):
        self.model.eval()
        self.validation_metrics.reset()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=description):
                lr_image, lr_masks, _, _ = self._prepare_batch(batch)

                pred_hr_masks_logits, _ = self.model(lr_image)
                predicted_hr_masks = torch.argmax(pred_hr_masks_logits, dim=1)    # (Batch, H, W)
                predicted_hr_masks = predicted_hr_masks.float()
                predicted_masks = self.transform.downsample_mask(predicted_hr_masks)
                predicted_masks = predicted_masks.long()

                self.validation_metrics.update(predicted_masks, lr_masks)

        return self.validation_metrics.compute()
    
    def _prepare_batch(self, batch):
        hr_image, hr_masks, lr_image, lr_masks = batch
        
        hr_image = hr_image.to(self.device, dtype=torch.float32)
        lr_image = lr_image.to(self.device, dtype=torch.float32)
        hr_masks = hr_masks.to(self.device, dtype=torch.long)
        lr_masks = lr_masks.to(self.device, dtype=torch.long)

        return lr_image, lr_masks, hr_image, hr_masks

    def get_primary_metric_name(self):
        return "dice"