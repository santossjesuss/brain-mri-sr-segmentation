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
# def visualize_seg_model(pipeline, dataset):
    # results_dict = pipeline.predict_random(dataset)
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