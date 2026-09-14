"""Fresh visual audit of exported assets, independent of persisted family edges."""

from __future__ import annotations

from agrivision_khaos.augmentation import CONFIRMED_LEVELS, candidate_pairs


def audit_visual_splits(paths, assignments, policy, cache, rejected_pairs=(), asset_digests=None):
    if set(paths) != set(assignments):
        raise ValueError("La auditoría requiere un split para cada activo exportado")
    descriptors = {key: cache.describe(path) for key, path in paths.items()}
    audit_policy = policy.model_copy(
        update={
            "candidate_neighbors": min(200, policy.candidate_neighbors * 2),
            "candidate_pool": min(8192, policy.candidate_pool * 2),
        }
    )
    metrics = {}
    candidates, saturated = candidate_pairs(descriptors, audit_policy, metrics)
    rejected = set(rejected_pairs)
    ambiguous = []
    confirmed = []
    checked = 0
    for left, right in candidates:
        if assignments[left] == assignments[right] or frozenset((left, right)) in rejected:
            continue
        checked += 1
        evidence = cache.verify(descriptors[left], descriptors[right], audit_policy)
        row = {"left": left, "right": right, **evidence}
        if evidence["level"] in CONFIRMED_LEVELS:
            confirmed.append(row)
        elif evidence["level"] == "candidate":
            ambiguous.append(row)
    if confirmed:
        examples = ", ".join(f"{row['left']} / {row['right']}" for row in confirmed[:5])
        raise RuntimeError(
            f"Fuga visual: {len(confirmed)} parejas confirmadas cruzan splits ({examples}); repite la detección y revisa las familias"
        )
    if asset_digests is not None:
        asset_digests.update({key: descriptor.asset for key, descriptor in descriptors.items()})
    return {
        "images": len(paths),
        "cross_split_pairs_checked": checked,
        "confirmed_cross_split_pairs": 0,
        "ambiguous_cross_split_pairs": ambiguous,
        "saturated_images": saturated,
        **metrics,
        "residual_risk": "Auditoría limitada por candidatos y transformaciones del detector; no demuestra ausencia de variantes no detectadas",
    }
