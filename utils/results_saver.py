import os
from torchvision.utils import save_image

def save_qualitative_results(experiments_dict, output_dir='qualitative_results'):
    os.makedirs(output_dir, exist_ok=True)

    lr_pipeline = experiments_dict['LR_Segmentation']
    hr_pipeline = experiments_dict['HR_Segmentation']

    input_image = lr_pipeline['input_image'].float()
    target_mask = hr_pipeline['target_mask'].float()

    save_image(input_image, os.path.join(output_dir, 'input_image.png'))
    save_image(target_mask, os.path.join(output_dir, 'ground_truth.png'))
    
    for name, results in experiments_dict.items():
        predicted_mask = results['predicted_mask'].float()
        save_image(predicted_mask, os.path.join(output_dir, f'{name.lower()}_prediction.png'))