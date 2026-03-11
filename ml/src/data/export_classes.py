"""Export class names into the JSON format used by runtime inference.

The deployed application expects a JSON file mapping integer output indices to
the original PlantVillage class labels. This script converts the split-time
``classes.txt`` artifact into that runtime-friendly representation.
"""

import argparse
import json
from pathlib import Path


def load_classes(classes_txt: Path) -> list[str]:
    """Read the ordered class list from ``classes.txt``.

    Args:
        classes_txt: Path to the newline-delimited class file.

    Returns:
        Ordered list of non-empty class labels.

    Raises:
        ValueError: If the file exists but contains no class labels.
    """
    classes = [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"No classes found in {classes_txt}")
    return classes


def main() -> None:
    """Create the runtime class-name JSON artifact.

    Side Effects:
        Writes the class-index mapping JSON expected by the deployed API.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes-txt", default="ml/splits/classes.txt")
    parser.add_argument("--out-json", default="ml/weights/classes.json")
    args = parser.parse_args()

    classes_txt = Path(args.classes_txt)
    out_json = Path(args.out_json)

    classes = load_classes(classes_txt)
    payload = {str(i): name for i, name in enumerate(classes)}

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(classes)} classes to {out_json}")


if __name__ == "__main__":
    main()
