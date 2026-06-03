import matplotlib.pyplot as plt

def visualize_superres_model(results_dict):
    input_image = results_dict['input_image']
    target_image = results_dict['target_image']
    predicted_image = results_dict['predicted_image']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(input_image.squeeze(), cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(target_image.squeeze(), cmap='gray')
    axes[1].set_title('Target Image')
    axes[1].axis('off')

    axes[2].imshow(predicted_image.squeeze(), cmap='gray')
    axes[2].set_title('Predicted Image')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_seg_model(results_dict):
    input_image = results_dict['input_image']
    target_mask = results_dict['target_mask']
    predicted_mask = results_dict['predicted_mask']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(input_image.squeeze(), cmap='gray')
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(target_mask.squeeze(), cmap='gray')
    axes[1].set_title('Target Mask')
    axes[1].axis('off')

    axes[2].imshow(predicted_mask.squeeze(), cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_all_experiments(experiments_dict):
    num_experiments = len(experiments_dict)
    fig, axes = plt.subplots(num_experiments, 3, figsize=(14, 3 * num_experiments))
    
    for i, (name, results) in enumerate(experiments_dict.items()):
        input_image = results['input_image']
        target_mask = results['target_mask']
        predicted_mask = results['predicted_mask']
        dice = results['dice']

        axes[i, 0].text(-0.2, 0.5, name, transform=axes[i, 0].transAxes, fontsize=12, weight='bold', va='center', ha='right')
        axes[i, 0].imshow(input_image.squeeze(), cmap='gray', aspect='equal')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(target_mask.squeeze(), cmap='gray', aspect='equal')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(predicted_mask.squeeze(), cmap='gray', aspect='equal')
        axes[i, 2].text(0.5, -0.05, f'Dice: {dice:.4f}', transform=axes[i, 2].transAxes, fontsize=10, ha='center', va='top')
        axes[i, 2].axis('off')

        if i == 0:
            axes[i, 0].set_title('Input Image')
            axes[i, 1].set_title('Target Mask')
            axes[i, 2].set_title('Predicted Mask')

    plt.tight_layout()
    plt.show()