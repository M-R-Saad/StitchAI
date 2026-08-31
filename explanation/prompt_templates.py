"""
Prompt templates for the explanation layer (Gemini Flash by default — swappable per
whitepaper Section 4.4). Explanation is a secondary annotation only: the AnomalyCLIP
score + heatmap remain the source of truth (Section 3.7).
"""

EXPLAIN_ANOMALY_TEMPLATE = """You are assisting a factory supervisor reviewing an automated
visual inspection result for the "{category}" category.

The system flagged this image as ANOMALOUS with confidence {score:.2f}.
The flagged region is roughly: {region_description}.

In 2-3 plain-language sentences, describe what likely looks wrong in the image, in terms
a non-technical line supervisor would understand. Do not invent specifics you can't see —
if uncertain, say so. Do not make a pass/fail recommendation; that decision is the
supervisor's.
"""


def build_explanation_prompt(category: str, score: float, region_description: str) -> str:
    return EXPLAIN_ANOMALY_TEMPLATE.format(
        category=category, score=score, region_description=region_description
    )
