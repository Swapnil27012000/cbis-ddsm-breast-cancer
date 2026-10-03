"""Data augmentation transforms for mammogram images."""
import torchvision.transforms as T
import torch

def get_train_transforms(image_size: tuple = (512, 512)):
    """Training transformations with benign/malignant invariant augmentations."""
    return T.Compose([
        T.ToPILImage(),
        T.Resize(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

def get_val_transforms(image_size: tuple = (512, 512)):
    """Validation and test transformations without random perturbations."""
    return T.Compose([
        T.ToPILImage(),
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
