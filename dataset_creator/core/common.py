"""
Common definitions and utilities shared across core modules.
"""

from enum import Enum


class Language(str, Enum):
    """Language for task and dataset generation"""
    ENGLISH = "english"
    CHINESE = "chinese" 