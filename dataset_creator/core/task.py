"""
Task module defines the Task class representing a specific AI task.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_serializer, model_validator

from .dataset import Dataset


class Task(BaseModel):
    """
    Represents a specific task for which to generate data and fine-tune models.
    
    A task defines what the model should learn to do, with instructions and requirements.
    """
    id: UUID = Field(default_factory=uuid4)
    name: str
    instruction: str
    description: Optional[str] = None
    project_id: Optional[UUID] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    system_prompt_template: Optional[str] = None
    thinking_instruction: Optional[str] = None
    datasets: Dict[UUID, Dataset] = Field(default_factory=dict)
    
    @field_serializer('created_at', 'updated_at')
    def serialize_datetime(self, dt: datetime) -> str:
        return dt.isoformat()
    
    @model_validator(mode='after')
    def set_default_system_prompt(self) -> 'Task':
        """Set default system prompt if not provided"""
        if not self.system_prompt_template:
            self.system_prompt_template = (
                "You are a helpful assistant that follows instructions carefully. "
                "Your task is to: {instruction}"
            )
        return self
    
    def create_dataset(
        self, 
        name: str, 
        model: str,
        questions: List[str],
        description: Optional[str] = None,
        provider: str = "openai"
    ) -> Dataset:
        """
        Create a new dataset for this task by generating responses to questions.
        
        Args:
            name: Name of the dataset
            model: Model to use for generating responses (e.g., 'gpt-4')
            questions: List of questions to include in the dataset
            description: Optional description of the dataset
            provider: Model provider (e.g., 'openai', 'anthropic')
            
        Returns:
            A new Dataset instance
        """
        from dataset_creator.data_gen import DataGenerator
        
        dataset = Dataset(
            name=name,
            description=description,
            task_id=self.id,
            model_used=model,
            provider=provider
        )
        
        # Create data generator
        data_generator = DataGenerator(model=model, provider=provider)
        
        # Generate examples
        examples = data_generator.generate_dataset(
            task=self,
            questions=questions
        )
        
        # Add examples to dataset
        for example in examples:
            dataset.add_example(example)
        
        # Store dataset
        self.datasets[dataset.id] = dataset
        self.updated_at = datetime.now()
        
        return dataset
    
    def get_dataset(self, dataset_id: Union[UUID, str]) -> Optional[Dataset]:
        """
        Get a dataset by ID.
        
        Args:
            dataset_id: ID of the dataset to get
            
        Returns:
            The Dataset if found, None otherwise
        """
        if isinstance(dataset_id, str):
            dataset_id = UUID(dataset_id)
        return self.datasets.get(dataset_id)
    
    def format_system_prompt(self) -> str:
        """
        Format the system prompt template with the task instruction.
        
        Returns:
            Formatted system prompt
        """
        return self.system_prompt_template.format(instruction=self.instruction) 