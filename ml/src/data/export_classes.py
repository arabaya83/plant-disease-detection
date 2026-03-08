"""Export class list to inference-friendly JSON index mapping."""

import argparse
import json
from pathlib import Path


def load_classes(classes_txt: Path) -> list[str]:
    """Read newline-separated class labels from split artifact."""
    classes = [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"No classes found in {classes_txt}")
    return classes


def main() -> None:
    """CLI entrypoint for creating `ml/weights/classes.json`."""
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
