"""Training/evaluation transform builders for PlantVillage images."""

from torchvision import transforms


def build_train_transforms(image_size: int):
    """Return standard training augmentations requested by project scope."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transforms(image_size: int):
    """Return deterministic preprocessing transforms for validation/testing."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
