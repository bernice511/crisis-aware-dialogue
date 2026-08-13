"""CRADLEBench label normalization and multi-hot encoding.

CRADLEBench's raw `final_labels` column is a comma-separated string of
`<crisis_type>_<ongoing|past>` tags, or `no_crisis`. The Layer 1 classifier
predicts only the 7 base crisis types (matching Layer 2's routing scheme and
CRADLE-Dialogue's crisis-type vocabulary); the ongoing/past temporal axis is
parsed and kept alongside each record but is not one of the model's outputs.

The normalization logic mirrors notebooks/eda.ipynb exactly (same raw-label
spelling inconsistencies: 'No crisis', 'suicideideation(passive)',
'childabuseendangerment'), so label statistics stay consistent between the EDA
and this pipeline.
"""

from __future__ import annotations

import re

# Order is fixed and is the model's output order everywhere in this package.
CRISIS_TYPES: tuple[str, ...] = (
    "selfharm",
    "suicideideation_passive",
    "suicideideation_active",
    "domesticviolence",
    "rape",
    "sexualharassment",
    "childabuse_endangerment",
)

_TEMPORAL_RE = re.compile(r"_(ongoing|past)$")


def normalize_label(label: str) -> str:
    """Collapse known raw-label spelling variants to a canonical form."""
    label = label.strip()
    if label.lower() in ("no crisis", "no_crisis"):
        return "no_crisis"
    label = label.replace("(passive)", "_passive").replace("(active)", "_active")
    label = label.replace("childabuseendangerment", "childabuse_endangerment")
    return label


def parse_labels(raw: object) -> list[str]:
    """Split a raw comma-separated `final_labels` cell into normalized labels."""
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []
    return [normalize_label(token) for token in text.split(",") if token.strip()]


def split_type_temporal(label: str) -> tuple[str, str]:
    """'domesticviolence_ongoing' -> ('domesticviolence', 'ongoing')."""
    if label == "no_crisis":
        return ("no_crisis", "none")
    match = _TEMPORAL_RE.search(label)
    if match:
        return (label[: match.start()], match.group(1))
    return (label, "unspecified")


def to_multihot(labels: list[str]) -> list[int]:
    """Multi-hot vector over CRISIS_TYPES, ignoring temporal and no_crisis."""
    present = {split_type_temporal(label)[0] for label in labels}
    return [1 if crisis_type in present else 0 for crisis_type in CRISIS_TYPES]


def temporal_by_type(labels: list[str]) -> dict[str, str]:
    """{'domesticviolence': 'ongoing', ...} for each crisis-type label present."""
    result: dict[str, str] = {}
    for label in labels:
        crisis_type, temporal = split_type_temporal(label)
        if crisis_type in CRISIS_TYPES:
            result[crisis_type] = temporal
    return result


def is_flagged(multihot: list[int]) -> bool:
    """True if any crisis type is present (the binary flagged/not-flagged view)."""
    return any(multihot)
