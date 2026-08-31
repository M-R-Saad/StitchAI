"""
Object-agnostic text prompts per RMG inspection category.

Per the whitepaper (Section 4.2), AnomalyCLIP's learned components are lightweight,
object-agnostic text prompts trained to separate generic 'normal' vs 'anomalous'
visual-semantic regions — NOT category-specific defect vocabulary (e.g. don't hardcode
"hole" vs "stain" as separate classes). Keep prompts generic; let the reference bank +
category selection do the adapting.

Add a new category by adding a new key here + a matching entry in config.yaml + a
reference_bank/<category>/ folder. Never fork model_wrapper.py per category.
"""

# Generic normal/anomalous phrasing, reused (with light category framing) across all
# categories — this is what keeps backbone/model_wrapper.py category-agnostic.
BASE_NORMAL_PROMPTS = [
    "a photo of a normal {category}",
    "a photo of a {category} without any defect",
    "a flawless {category}",
]

BASE_ANOMALOUS_PROMPTS = [
    "a photo of a {category} with an anomaly",
    "a photo of a damaged {category}",
    "a {category} with a visible defect",
]

CATEGORY_PROMPTS = {
    "fabric": {
        "object_name": "fabric",
        # TODO (Phase 1): validate/tune wording against the fabric-defect dataset you pick.
    },
    "safety": {
        "object_name": "worker safety scene",
        # TODO (Phase 3): validate against the PPE dataset. Keep it about the *scene*
        # being compliant/non-compliant rather than naming specific PPE items, to stay
        # consistent with the object-agnostic design.
    },
    "machinery": {
        "object_name": "machine component",
        # TODO (Phase 4): validate against MVTec-AD proxy subset. Remember: proof of
        # concept only — label as such in the UI (whitepaper Section 3.2).
    },
}


def get_prompts(category: str):
    """Return (normal_prompts, anomalous_prompts) for a given category."""
    if category not in CATEGORY_PROMPTS:
        raise ValueError(f"Unknown category: {category!r}. Known: {list(CATEGORY_PROMPTS)}")
    object_name = CATEGORY_PROMPTS[category]["object_name"]
    normal = [p.format(category=object_name) for p in BASE_NORMAL_PROMPTS]
    anomalous = [p.format(category=object_name) for p in BASE_ANOMALOUS_PROMPTS]
    return normal, anomalous
