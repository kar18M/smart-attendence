"""
face_recognition_module/encodings_store.py
-------------------------------------------
In-memory cache of known face encodings loaded from MongoDB.

Call ``refresh()`` after enrolling new students so the live video processor
picks up the change without restarting.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_operations import get_all_students

logger = logging.getLogger(__name__)


class EncodingsStore:
    def __init__(self) -> None:
        self.encodings: List[np.ndarray] = []
        self.metadata: List[dict] = []
        self._loaded = False

    def refresh(self) -> int:
        students = get_all_students()
        self.encodings = [s["face_encoding"] for s in students]
        self.metadata = [
            {"student_id": s["student_id"], "name": s["name"]}
            for s in students
        ]
        self._loaded = True
        logger.info("EncodingsStore refreshed: %d student(s) loaded.", len(students))
        return len(students)

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.refresh()

    @property
    def is_empty(self) -> bool:
        self.ensure_loaded()
        return len(self.encodings) == 0


_store: EncodingsStore | None = None


def get_store() -> EncodingsStore:
    global _store
    if _store is None:
        _store = EncodingsStore()
    return _store
