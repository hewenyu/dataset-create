"""
Dataset module defines the Dataset class and related models for representing training data.
"""

import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_serializer


class DatasetFormat(str, Enum):
    """Format of the dataset for fine-tuning"""
    CHAT_JSONL = "chat_jsonl"
    INSTRUCTION_JSONL = "instruction_jsonl"
    CUSTOM_JSONL = "custom_jsonl"


class DatasetSplit(BaseModel):
    """
    Represents a split of a dataset (e.g., train, validation, test).
    """
    id: UUID = Field(default_factory=uuid4)
    name: str
    examples: List[UUID] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @field_serializer('created_at')
    def serialize_datetime(self, dt: datetime) -> str:
        return dt.isoformat()


class DatasetExample(BaseModel):
    """
    Represents a single example in a dataset (question/answer pair).
    """
    id: UUID = Field(default_factory=uuid4)
    question: str
    answer: str
    system_prompt: str
    model_used: str
    thinking: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @field_serializer('created_at')
    def serialize_datetime(self, dt: datetime) -> str:
        return dt.isoformat()
    
    def to_chat_format(self) -> Dict:
        """
        Convert the example to OpenAI chat format.
        
        Returns:
            Dict in OpenAI chat format
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.question}
        ]
        
        # Add thinking as assistant message if available
        if self.thinking:
            messages.append({"role": "assistant", "content": self.thinking})
            
        # Add final answer as assistant message
        messages.append({"role": "assistant", "content": self.answer})
        
        return {
            "messages": messages
        }
    
    def to_instruction_format(self) -> Dict:
        """
        Convert the example to instruction fine-tuning format.
        
        Returns:
            Dict in instruction format (instruction/response)
        """
        instruction = f"System: {self.system_prompt}\n\nUser: {self.question}"
        
        return {
            "instruction": instruction,
            "response": self.answer
        }


class Dataset(BaseModel):
    """
    Represents a dataset for fine-tuning language models.
    """
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: Optional[str] = None
    task_id: Optional[UUID] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    model_used: str
    provider: str
    examples: Dict[UUID, DatasetExample] = Field(default_factory=dict)
    splits: Dict[str, DatasetSplit] = Field(default_factory=dict)
    directory: Optional[Path] = None
    
    @field_serializer('created_at', 'updated_at')
    def serialize_datetime(self, dt: datetime) -> str:
        return dt.isoformat()
        
    @field_serializer('directory')
    def serialize_path(self, path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        return str(path)
    
    def add_example(self, example: DatasetExample) -> UUID:
        """
        Add an example to the dataset.
        
        Args:
            example: The example to add
            
        Returns:
            UUID of the added example
        """
        self.examples[example.id] = example
        self.updated_at = datetime.now()
        
        # Add to default split if it exists
        if "train" in self.splits:
            self.splits["train"].examples.append(example.id)
        
        return example.id
    
    def create_split(self, name: str, example_ids: Optional[List[UUID]] = None) -> DatasetSplit:
        """
        Create a new split in the dataset.
        
        Args:
            name: Name of the split
            example_ids: Optional list of example IDs to include in the split
            
        Returns:
            The created DatasetSplit
        """
        split = DatasetSplit(name=name)
        
        if example_ids:
            # Verify all examples exist
            for example_id in example_ids:
                if example_id not in self.examples:
                    raise ValueError(f"Example {example_id} not found in dataset")
            split.examples = example_ids
            
        self.splits[name] = split
        self.updated_at = datetime.now()
        
        return split
    
    def get_example(self, example_id: Union[UUID, str]) -> Optional[DatasetExample]:
        """
        Get an example by ID.
        
        Args:
            example_id: ID of the example to get
            
        Returns:
            The example if found, None otherwise
        """
        if isinstance(example_id, str):
            example_id = UUID(example_id)
        return self.examples.get(example_id)
    
    def save(self, directory: Optional[Path] = None, format: DatasetFormat = DatasetFormat.CHAT_JSONL) -> Path:
        """
        Save the dataset to disk.
        
        Args:
            directory: Directory to save the dataset to. If None, uses self.directory or creates a new one.
            format: Format to save the dataset in
            
        Returns:
            Path to the saved dataset directory
        """
        if directory is not None:
            self.directory = directory
        
        if self.directory is None:
            # Create a directory based on dataset name if not specified
            self.directory = Path(f"./datasets/{self.name.replace(' ', '_').lower()}")
        
        # Create directory if it doesn't exist
        os.makedirs(self.directory, exist_ok=True)
        
        # Save dataset metadata
        dataset_file = self.directory / "dataset.json"
        with open(dataset_file, "w") as f:
            # Exclude examples, splits and directory from serialization
            dataset_data = self.model_dump(exclude={"examples", "splits", "directory"})
            json.dump(dataset_data, f, indent=2)
        
        # Save examples
        examples_dir = self.directory / "examples"
        os.makedirs(examples_dir, exist_ok=True)
        
        for example_id, example in self.examples.items():
            example_file = examples_dir / f"{example_id}.json"
            with open(example_file, "w") as f:
                json.dump(example.model_dump(), f, indent=2)
        
        # Save splits
        splits_dir = self.directory / "splits"
        os.makedirs(splits_dir, exist_ok=True)
        
        for split_name, split in self.splits.items():
            split_file = splits_dir / f"{split_name}.json"
            with open(split_file, "w") as f:
                json.dump(split.model_dump(), f, indent=2)
        
        # Save the dataset in the specified format
        self._save_formatted(format)
        
        return self.directory
    
    def _save_formatted(self, format: DatasetFormat) -> None:
        """
        Save the dataset in the specified format.
        
        Args:
            format: Format to save the dataset in
        """
        if not self.directory:
            raise ValueError("Dataset directory must be set before saving formatted dataset")
        
        # Create exports directory
        exports_dir = self.directory / "exports"
        os.makedirs(exports_dir, exist_ok=True)
        
        # Save full dataset
        self._save_split_formatted(format, "full", list(self.examples.keys()))
        
        # Save each split
        for split_name, split in self.splits.items():
            self._save_split_formatted(format, split_name, split.examples)
    
    def _save_split_formatted(self, format: DatasetFormat, split_name: str, example_ids: List[UUID]) -> None:
        """
        Save a split of the dataset in the specified format.
        
        Args:
            format: Format to save the dataset in
            split_name: Name of the split
            example_ids: IDs of examples to include in the export
        """
        if not self.directory:
            raise ValueError("Dataset directory must be set before saving formatted dataset")
        
        exports_dir = self.directory / "exports"
        formatted_file = exports_dir / f"{split_name}_{format.value}.jsonl"
        
        with open(formatted_file, "w") as f:
            for example_id in example_ids:
                example = self.examples.get(example_id)
                if example:
                    if format == DatasetFormat.CHAT_JSONL:
                        formatted = example.to_chat_format()
                    elif format == DatasetFormat.INSTRUCTION_JSONL:
                        formatted = example.to_instruction_format()
                    else:  # Custom format - use the raw example
                        formatted = example.model_dump()
                    
                    f.write(json.dumps(formatted) + "\n")
    
    @classmethod
    def load(cls, directory: Union[str, Path]) -> "Dataset":
        """
        Load a dataset from disk.
        
        Args:
            directory: Directory to load the dataset from
            
        Returns:
            Loaded Dataset instance
        """
        if isinstance(directory, str):
            directory = Path(directory)
        
        # Load dataset metadata
        dataset_file = directory / "dataset.json"
        with open(dataset_file, "r") as f:
            dataset_data = json.load(f)
        
        # Create dataset instance
        dataset = cls(**dataset_data, directory=directory)
        
        # Load examples
        examples_dir = directory / "examples"
        if examples_dir.exists():
            for example_file in examples_dir.glob("*.json"):
                with open(example_file, "r") as f:
                    example_data = json.load(f)
                    example = DatasetExample(**example_data)
                    dataset.examples[example.id] = example
        
        # Load splits
        splits_dir = directory / "splits"
        if splits_dir.exists():
            for split_file in splits_dir.glob("*.json"):
                with open(split_file, "r") as f:
                    split_data = json.load(f)
                    split = DatasetSplit(**split_data)
                    dataset.splits[split.name] = split
        
        return dataset 