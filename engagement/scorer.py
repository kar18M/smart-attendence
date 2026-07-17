"""
engagement/scorer.py
---------------------
Maps emotion labels to a three-tier engagement score.

Mapping rationale
-----------------
* Engaged    — student is attentive / positive: happy, neutral
* Disengaged — student is struggling / withdrawn: sad, fear, disgust
* Neutral    — ambiguous or mildly off-task: angry, surprise
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_EMOTION_TO_SCORE: dict[str, str] = {
    "happy":    "Engaged",
    "neutral":  "Engaged",
    "sad":      "Disengaged",
    "fear":     "Disengaged",
    "disgust":  "Disengaged",
    "angry":    "Neutral",
    "surprise": "Neutral",
}

VALID_SCORES = ("Engaged", "Neutral", "Disengaged")
SCORE_COLOURS = {
    "Engaged":    "#2ecc71",
    "Neutral":    "#f39c12",
    "Disengaged": "#e74c3c",
}


def score_engagement(emotion: str) -> str:
    score = _EMOTION_TO_SCORE.get(emotion.lower(), "Neutral")
    logger.debug("emotion=%s → score=%s", emotion, score)
    return score


def score_to_numeric(score: str) -> float:
    mapping = {"Engaged": 2.0, "Neutral": 1.0, "Disengaged": 0.0}
    return mapping.get(score, 1.0)


def compute_session_stats(scores: list[str]) -> dict:
    if not scores:
        return {"total": 0, "engaged": 0, "neutral": 0, "disengaged": 0, "pct_engaged": 0.0}
    total = len(scores)
    engaged = scores.count("Engaged")
    neutral = scores.count("Neutral")
    disengaged = scores.count("Disengaged")
    return {
        "total": total,
        "engaged": engaged,
        "neutral": neutral,
        "disengaged": disengaged,
        "pct_engaged": round(engaged / total * 100, 1),
    }
