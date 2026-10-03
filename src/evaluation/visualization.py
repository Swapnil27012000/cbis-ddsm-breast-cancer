"""Mammogram prediction visualization."""
import matplotlib.pyplot as plt
import numpy as np

def visualize_mammogram_predictions(images: list, true_labels: list, pred_labels: list, save_path: str = None):
    """Plot a grid of sample mammograms with true and predicted labels."""
    n = min(len(images), 8)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    classes = ["Benign", "Malignant"]
    for i in range(n):
        ax = axes[i // 4, i % 4]
        ax.imshow(images[i], cmap="gray")
        title = f"True: {classes[true_labels[i]]}\nPred: {classes[pred_labels[i]]}"
        color = "green" if true_labels[i] == pred_labels[i] else "red"
        ax.set_title(title, color=color)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
