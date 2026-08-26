import os
from torchvision.utils import save_image

def save_qualitative_results(experiments_dict, output_dir='qualitative_results'):
    os.makedirs(output_dir, exist_ok=True)
    
    for name, results in experiments_dict.items():
        input_image = results['input_image'].float()
        target_mask = results['target_mask'].float()
        predicted_mask = results['predicted_mask'].float()

        save_image(input_image, os.path.join(output_dir, f'{name}_image.png'))
        save_image(target_mask, os.path.join(output_dir, f'{name}_target.png'))
        save_image(predicted_mask, os.path.join(output_dir, f'{name}_prediction.png'))