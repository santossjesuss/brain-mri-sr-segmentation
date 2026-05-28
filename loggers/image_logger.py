import os
import numpy as np

class ImageLogger():
    def __init__(self, base_img_log_dir, model_name):
        self.log_dir = os.path.join(base_img_log_dir, model_name)
        os.makedirs(self.log_dir, exist_ok=True)

    def log_batch_images(self, batch_images, epoch, batch_idx):
        directory = os.path.join(self.log_dir, f'epoch_{epoch}')
        os.makedirs(directory, exist_ok=True)
        
        for idx, img in enumerate(batch_images):
            img = img.detach().cpu().numpy()
            contains_nan = np.isnan(img).any()
            if contains_nan:
                print(f"Warning: Image at batch {batch_idx}, index {idx} contains NaN values.")
                filename = f'batch_{batch_idx}_{idx}_contains_nan.npy'
            else:
                filename = f'batch_{batch_idx}_{idx}.png'
            
            filepath = os.path.join(directory, filename)
            np.save(filepath, img)