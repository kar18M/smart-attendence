"""
database/mongo_client.py
------------------------
Provides a single, cached MongoDB connection for the whole application.

Usage:
    from database.mongo_client import get_db

    db = get_db()            # raises MongoConnectionError if MongoDB is down
    collection = db["students"]
"""

from __future__ import annotations

import logging

import pymongo
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class MongoConnectionError(RuntimeError):
    """Raised when the local MongoDB instance is not reachable."""


_client: MongoClient | None = None
_db = None


def get_db():
    """
    Return the ``smart_attendance`` MongoDB database object.
    """
    global _client, _db

    if _db is not None:
        return _db

    try:
        _client = MongoClient(
            config.MONGO_URI,
            serverSelectionTimeoutMS=3000,
        )
        _client.admin.command("ping")
    except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
        _client = None
        raise MongoConnectionError(
            f"Cannot connect to MongoDB at {config.MONGO_URI}. "
            "Make sure the MongoDB service is running:\n\n"
            "  sudo systemctl start mongod\n"
            "or\n"
            "  mongod --dbpath /data/db\n\n"
            f"Original error: {exc}"
        ) from exc

    _db = _client[config.DB_NAME]
    _ensure_indexes(_db)
    logger.info("Connected to MongoDB database '%s'", config.DB_NAME)
    return _db


def _ensure_indexes(db) -> None:
    db[config.COL_ATTENDANCE].create_index(
        [("student_id", pymongo.ASCENDING), ("date", pymongo.ASCENDING)],
        unique=False,
        name="attendance_student_date_idx",
        background=True,
    )
    db[config.COL_ENGAGEMENT].create_index(
        [("student_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)],
        name="engagement_student_ts_idx",
        background=True,
    )
    db[config.COL_STUDENTS].create_index(
        [("student_id", pymongo.ASCENDING)],
        unique=True,
        name="students_id_unique_idx",
        background=True,
    )
    logger.debug("MongoDB indexes ensured.")


def close_db() -> None:
    """Close the MongoDB client connection (useful in test teardown)."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None
