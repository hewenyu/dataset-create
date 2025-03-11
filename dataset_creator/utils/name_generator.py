"""
Name generator utility for creating unique, memorable names.
"""

import random
from datetime import datetime
from typing import List, Optional


# Lists of words for generating memorable names
ADJECTIVES = [
    "autumn", "hidden", "bitter", "misty", "silent", "empty", "dry", "dark",
    "summer", "icy", "delicate", "quiet", "white", "cool", "spring", "winter",
    "patient", "twilight", "dawn", "crimson", "wispy", "weathered", "blue",
    "billowing", "broken", "cold", "damp", "falling", "frosty", "green",
    "long", "late", "lingering", "bold", "little", "morning", "muddy", "old",
    "red", "rough", "still", "small", "sparkling", "throbbing", "shy",
    "wandering", "withered", "wild", "black", "young", "holy", "solitary",
    "fragrant", "aged", "snowy", "proud", "floral", "restless", "divine",
    "polished", "ancient", "purple", "lively", "nameless"
]

NOUNS = [
    "waterfall", "river", "breeze", "moon", "rain", "wind", "sea", "morning",
    "snow", "lake", "sunset", "pine", "shadow", "leaf", "dawn", "glitter",
    "forest", "hill", "cloud", "meadow", "sun", "glade", "bird", "brook",
    "butterfly", "bush", "dew", "dust", "field", "fire", "flower", "firefly",
    "feather", "grass", "haze", "mountain", "night", "pond", "darkness",
    "snowflake", "silence", "sound", "sky", "shape", "surf", "thunder",
    "violet", "water", "wildflower", "wave", "water", "resonance", "sun",
    "wood", "dream", "cherry", "tree", "fog", "frost", "voice", "paper",
    "frog", "smoke", "star"
]


def generate_unique_name(
    adjectives: Optional[List[str]] = None,
    nouns: Optional[List[str]] = None,
    separator: str = "-",
    include_timestamp: bool = False
) -> str:
    """
    Generate a unique, memorable name.
    
    Args:
        adjectives: Optional list of adjectives to use
        nouns: Optional list of nouns to use
        separator: Separator between words
        include_timestamp: Whether to include a timestamp for uniqueness
        
    Returns:
        A unique, memorable name
    """
    adj_list = adjectives or ADJECTIVES
    noun_list = nouns or NOUNS
    
    adj = random.choice(adj_list)
    noun = random.choice(noun_list)
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{adj}{separator}{noun}{separator}{timestamp}"
    else:
        return f"{adj}{separator}{noun}" 