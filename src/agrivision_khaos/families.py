"""Stable identities and union of all split constraints, including excluded bridges."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from agrivision_khaos.augmentation import CONFIRMED_LEVELS


def value(sample, name, default=None):
    if isinstance(sample, dict):
        return sample.get(name, default)
    if hasattr(sample, "has_field"):
        return sample.get_field(name) if sample.has_field(name) else default
    return getattr(sample, name, default)


def stable_identity(sample) -> str:
    payload = [
        value(sample, "asset_sha256", ""),
        value(sample, "source_dataset", ""),
        value(sample, "source_path") or value(sample, "filepath", ""),
    ]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


class Union:
    def __init__(self, ids):
        self.parent = {key: key for key in ids}

    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, left, right):
        if left not in self.parent or right not in self.parent:
            return
        a, b = self.find(left), self.find(right)
        self.parent[max(a, b)] = min(a, b)


def relation_groups(samples, *, confirmed_only=False, captures=False, locations=False):
    samples = list(samples)
    by_id = {sample.id: sample for sample in samples}
    union = Union(by_id)
    rejected = human_rejections(samples)
    tokens = {}
    for sample in samples:
        for link in value(sample, "duplicate_links", []) or []:
            if link.get("rejected") or frozenset((sample.id, link["other_id"])) in rejected:
                continue
            if confirmed_only and link.get("level") not in CONFIRMED_LEVELS:
                continue
            union.join(sample.id, link["other_id"])
        fields = []
        if captures:
            fields = ["capture_group", "video_id"] + (["location"] if locations else [])
            # Legacy datasets have only a cluster field. New datasets derive it from links.
            if not value(sample, "duplicate_schema_version"):
                fields.append("duplicate_cluster_id")
        for name in fields:
            field = value(sample, name)
            if field:
                scope = (
                    "" if name == "duplicate_cluster_id" else value(sample, "source_dataset", "")
                )
                token = (name, scope, str(field))
                if token in tokens:
                    union.join(sample.id, tokens[token])
                else:
                    tokens[token] = sample.id
    groups = defaultdict(list)
    for key in by_id:
        groups[union.find(key)].append(key)
    assignments = {}
    for members in groups.values():
        identity = "\0".join(sorted({stable_identity(by_id[key]) for key in members}))
        group_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
        for key in members:
            assignments[key] = group_id
    return assignments


def human_rejections(samples):
    return {
        frozenset((sample.id, link["other_id"]))
        for sample in samples
        for link in value(sample, "duplicate_links", []) or []
        if link.get("method") == "human_review" and link.get("rejected")
    }


def audit_assignments(samples, assignments, *, locations=False):
    groups = relation_groups(samples, captures=True, locations=locations)
    splits = defaultdict(set)
    for key, split in assignments.items():
        splits[groups[key]].add(split)
    conflicts = [key for key, values in splits.items() if len(values) > 1]
    if conflicts:
        raise RuntimeError(f"Fuga entre particiones: {len(conflicts)} grupos divididos")
    return {"checked_samples": len(assignments), "groups": len(splits), "conflicts": 0}
