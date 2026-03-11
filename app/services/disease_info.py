"""Label parsing and user-facing disease descriptions.

This module converts raw model class labels such as ``Tomato___Late_blight``
into frontend-friendly crop names, disease names, and educational text. It is
kept separate from model inference so label presentation can evolve without
changing classifier code.
"""

from typing import Dict, Tuple


DISEASE_DESCRIPTIONS: Dict[str, str] = {
    "Apple___Apple_scab": "Fungal disease causing olive-green lesions on leaves and fruit.",
    "Apple___Black_rot": "Fungal infection that causes leaf spots and fruit rot.",
    "Apple___Cedar_apple_rust": "Rust disease producing orange/yellow lesions on leaves.",
    "Apple___healthy": "Healthy apple leaf with no visible disease symptoms.",
    "Tomato___Late_blight": "Serious oomycete disease with dark lesions and rapid spread.",
    "Tomato___Early_blight": "Fungal disease creating concentric ring spots on lower leaves.",
    "Tomato___Leaf_Mold": "Fungal disease with yellow patches and mold on leaf undersides.",
    "Tomato___healthy": "Healthy tomato leaf with no visible disease symptoms.",
    "Potato___Late_blight": "Fast-spreading disease causing water-soaked dark lesions.",
    "Potato___Early_blight": "Fungal disease with target-like spots and leaf yellowing.",
    "Potato___healthy": "Healthy potato leaf with no visible disease symptoms.",
}


def parse_label(class_label: str) -> Tuple[str, str, str]:
    """Convert a raw dataset label into user-facing fields.

    Args:
        class_label: Raw class label in PlantVillage naming format.

    Returns:
        A tuple of ``(crop_name, disease_name, health_status)``.
    """
    if "___" in class_label:
        crop, disease = class_label.split("___", 1)
    else:
        crop, disease = "Unknown", class_label

    if disease.lower() == "healthy":
        return crop, "No disease detected", "Healthy"
    return crop, disease.replace("_", " "), "Diseased"


def get_description(class_label: str, healthy_or_diseased: str) -> str:
    """Return a short educational description for a predicted label.

    Args:
        class_label: Raw model label for the predicted class.
        healthy_or_diseased: High-level health state derived from the label.

    Returns:
        A concise user-facing message suitable for the result card UI.
    """
    if healthy_or_diseased == "Healthy":
        return "Healthy - No disease detected."
    return DISEASE_DESCRIPTIONS.get(
        class_label,
        "Leaf pattern suggests disease. Confirm with local agronomy guidance.",
    )
