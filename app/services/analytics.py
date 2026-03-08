"""Lightweight analytics logging and aggregation service."""

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


class AnalyticsService:
    """Persists inference events and computes aggregate summary statistics."""

    def __init__(self, analytics_log_path: str):
        self.log_path = Path(analytics_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, record: Dict[str, Any]) -> None:
        """Append one JSON event to analytics log."""
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def summary(self) -> Dict[str, Any]:
        """Compute diagnosis volume, quality, confidence, and latency aggregates."""
        if not self.log_path.exists():
            return {
                "total_diagnoses": 0,
                "invalid_image_rate": 0.0,
                "average_confidence": 0.0,
                "common_diseases": [],
                "average_response_time_ms": 0.0,
            }

        rows: List[Dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        if not rows:
            return {
                "total_diagnoses": 0,
                "invalid_image_rate": 0.0,
                "average_confidence": 0.0,
                "common_diseases": [],
                "average_response_time_ms": 0.0,
            }

        total = len(rows)
        invalid = sum(1 for r in rows if r.get("status") != "ok")
        confidences = [
            leaf["confidence"]
            for r in rows
            for leaf in r.get("results", [])
            if isinstance(leaf.get("confidence"), (int, float))
        ]
        latencies = [r.get("latency_ms", 0.0) for r in rows if isinstance(r.get("latency_ms"), (int, float))]
        disease_counter = Counter(
            leaf.get("disease_name", "Unknown")
            for r in rows
            for leaf in r.get("results", [])
            if leaf.get("healthy_or_diseased") == "Diseased"
        )

        return {
            "total_diagnoses": total,
            "invalid_image_rate": round(invalid / total, 4),
            "average_confidence": round(mean(confidences), 4) if confidences else 0.0,
            "common_diseases": disease_counter.most_common(5),
            "average_response_time_ms": round(mean(latencies), 2) if latencies else 0.0,
        }
