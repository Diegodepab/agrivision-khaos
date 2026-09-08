"""Append-only decision snapshots for inspection and explicit rollback."""

from __future__ import annotations

from datetime import UTC, datetime

import fiftyone as fo


def ensure_history(dataset):
    if "curation_history" not in dataset.get_field_schema():
        dataset.add_sample_field("curation_history", fo.ListField, subfield=fo.DictField)


def snapshot(sample):
    current = sample.get_field("curation") if sample.has_field("curation") else None
    return {
        "curation": {key: value for key, value in current.to_dict().items() if key != "_cls"}
        if current is not None
        else {},
        "tags": list(sample.tags),
    }


def record_transition(sample, before, action):
    after = snapshot(sample)
    if before == after:
        return
    history = list(sample.get_field("curation_history") or [])
    history.append(
        {"at": datetime.now(UTC).isoformat(), "action": action, "before": before, "after": after}
    )
    sample["curation_history"] = history
