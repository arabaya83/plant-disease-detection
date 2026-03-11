"""Runtime analytics logging and aggregation helpers.

The deployed API records one JSON object per inference request in a JSONL log.
This module keeps logging and summary calculations in one place so the route
handlers remain focused on request orchestration.
"""

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


class AnalyticsService:
    """Persist inference events and compute aggregate usage statistics."""

    def __init__(self, analytics_log_path: str):
        """Prepare the analytics log destination.

        Args:
            analytics_log_path: Path to the JSONL file used to store one event
                per inference request.
        """
        self.log_path = Path(analytics_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, record: Dict[str, Any]) -> None:
        """Append one inference event to the JSONL analytics log.

        Args:
            record: JSON-serializable analytics payload.

        Side Effects:
            Appends one line to the analytics log file.
        """
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def summary(self) -> Dict[str, Any]:
        """Compute aggregate analytics from the stored JSONL log.

        Returns:
            Summary metrics suitable for dashboards or inspection, including
            total diagnoses, invalid-rate, confidence averages, common diseases,
            and average latency.
        """
        if not self.log_path.exists():
            return {
                "total_diagnoses": 0,
                "invalid_image_rate": 0.0,
                "average_confidence": 0.0,
                "common_diseases": [],
                "average_response_time_ms": 0.0,
            }

        records: List[Dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            return {
                "total_diagnoses": 0,
                "invalid_image_rate": 0.0,
                "average_confidence": 0.0,
                "common_diseases": [],
                "average_response_time_ms": 0.0,
            }

        total = len(records)
        invalid = sum(1 for record in records if record.get("status") != "ok")
        confidences = [
            leaf["confidence"]
            for record in records
            for leaf in record.get("results", [])
            if isinstance(leaf.get("confidence"), (int, float))
        ]
        latencies = [
            record.get("latency_ms", 0.0)
            for record in records
            if isinstance(record.get("latency_ms"), (int, float))
        ]
        disease_counter = Counter(
            leaf.get("disease_name", "Unknown")
            for record in records
            for leaf in record.get("results", [])
            if leaf.get("healthy_or_diseased") == "Diseased"
        )

        return {
            "total_diagnoses": total,
            "invalid_image_rate": round(invalid / total, 4),
            "average_confidence": round(mean(confidences), 4) if confidences else 0.0,
            "common_diseases": disease_counter.most_common(5),
            "average_response_time_ms": round(mean(latencies), 2) if latencies else 0.0,
        }
