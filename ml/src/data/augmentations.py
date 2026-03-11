"""Image transform builders shared across training and evaluation.

Keeping transforms in one module helps the project document which
preprocessing steps are part of training augmentation and which are part of the
deterministic evaluation/inference path.
"""

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transforms(image_size: int) -> transforms.Compose:
    """Build the training augmentation pipeline.

    Args:
        image_size: Final square resolution expected by the model.

    Returns:
        A torchvision ``Compose`` object with resizing, augmentation, tensor
        conversion, and ImageNet-style normalization.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transforms(image_size: int) -> transforms.Compose:
    """Build deterministic transforms for validation, test, and inference.

    Args:
        image_size: Final square resolution expected by the model.

    Returns:
        A torchvision ``Compose`` object without stochastic augmentations.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
