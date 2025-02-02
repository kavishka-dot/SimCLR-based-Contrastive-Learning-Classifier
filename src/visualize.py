import matplotlib.pyplot as plt
import numpy as np
import torch

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Function to show images with predicted and true labels
def visualize_predictions(model, test_loader, device, num_images=10):
    '''Visualize images with predicted and true labels'''
    model.eval()
    images, true_labels, pred_labels = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)

            images.extend(x.cpu())  # Store images
            true_labels.extend(y.cpu().numpy())  # Store true labels
            pred_labels.extend(predicted.cpu().numpy())  # Store predicted labels

            if len(images) >= num_images:
                break  # Stop when we have enough images

    fig, axes = plt.subplots(2, num_images // 2, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(num_images):
        img = images[i].permute(1, 2, 0).numpy()  # Convert from tensor to image
        img = (img * 0.5) + 0.5  # Unnormalize

        true_label = class_names[true_labels[i]]
        pred_label = class_names[pred_labels[i]]

        # Choose color: Green if correct, Red if wrong
        color = "green" if true_label == pred_label else "red"

        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"Pred: {pred_label}", color=color)
        axes[i].set_xlabel(f"True: {true_label}")

    plt.tight_layout()
    plt.show()