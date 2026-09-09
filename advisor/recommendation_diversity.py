"""Conservative grouping after grounded review, never before market scanning."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence


def select_diverse_recommendations(
    plugin_ids: Sequence[str],
    reviews: Mapping[str, dict],
    *,
    enabled: bool = True,
    tolerance: int = 3,
    profiles: Mapping[str, dict | None] | None = None,
    capability_counts: Mapping[int, int] | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    """Keep the first N score-ranked alternatives per verified capability set.

    Only fully supported, identical capability sets are interchangeable here.
    Missing/partial/unknown evidence never proves equivalence. Additional
    supported capabilities form a distinct group, preserving complementary uses.
    IDs and titles alone never provide evidence of functional duplication.
    """
    if not enabled:
        return list(plugin_ids), {}
    tolerance = max(1, min(20, tolerance))
    groups: dict[tuple, list[str]] = defaultdict(list)
    selected: list[str] = []
    folded: dict[str, list[str]] = defaultdict(list)
    for pid in plugin_ids:
        review = reviews.get(pid, {})
        checks = review.get("capability_checks", [])
        supported = []
        uncertain = not checks
        for check in checks:
            if check.get("status") in ("partial", "unknown"):
                uncertain = True
            if check.get("status") == "supported":
                supported.append((check["need_index"], check["capability_index"]))
        signature = tuple(sorted(set(supported)))
        if checks and capability_counts:
            provided = {(c['need_index'], c['capability_index']) for c in checks}
            uncertain |= any(
                (ni, ci) not in provided
                for ni in {ni for ni, _ in supported}
                for ci in range(1, capability_counts.get(ni, 0) + 1)
            )
        if not checks:
            # Local retrieval uses the older review contract. Only identical,
            # high-confidence profiles with no limitations are a safe fallback.
            profile = (profiles or {}).get(pid) or {}
            caps = tuple(sorted({str(c).strip().casefold() for c in profile.get('capabilities', []) if str(c).strip()}))
            if (caps and profile.get('confidence', 0) >= 0.75
                    and not profile.get('limitations') and not review.get('risks')
                    and review.get('functional_fit', 0) > 0):
                signature = ('profile', caps, tuple(sorted(review.get('matched_need_titles', []))))
                uncertain = False
        if uncertain or not signature:
            selected.append(pid)
            continue
        group = groups[signature]
        if len(group) < tolerance:
            group.append(pid)
            selected.append(pid)
        else:
            folded[group[0]].append(pid)
    return selected, dict(folded)
