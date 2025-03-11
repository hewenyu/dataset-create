"""
File utility functions for the dataset creator.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to ensure exists
        
    Returns:
        Path object for the directory
    """
    if isinstance(directory, str):
        directory = Path(directory)
    
    os.makedirs(directory, exist_ok=True)
    return directory


def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dictionaries.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line in the file
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Save a list of dictionaries to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to save the JSONL file to
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    # Ensure parent directory exists
    ensure_directory(file_path.parent)
    
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n") 