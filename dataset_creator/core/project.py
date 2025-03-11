"""
Project module defines the DatasetProject class which manages tasks and datasets
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_serializer

from .task import Task


class DatasetProject(BaseModel):
    """
    Represents a project that contains tasks and datasets for fine-tuning LLMs.
    
    A project is the top-level container for organizing related tasks and datasets.
    """
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    tasks: Dict[UUID, Task] = Field(default_factory=dict)
    directory: Optional[Path] = None
    
    @field_serializer('created_at', 'updated_at')
    def serialize_datetime(self, dt: datetime) -> str:
        return dt.isoformat()

    @field_serializer('directory')
    def serialize_path(self, path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        return str(path)
    
    def create_task(self, name: str, instruction: str, description: Optional[str] = None) -> Task:
        """
        Create a new task within the project.
        
        Args:
            name: The name of the task
            instruction: The instruction for the task
            description: Optional description of the task
            
        Returns:
            A new Task instance
        """
        task = Task(
            name=name,
            instruction=instruction, 
            description=description,
            project_id=self.id
        )
        self.tasks[task.id] = task
        self.updated_at = datetime.now()
        return task
    
    def get_task(self, task_id: Union[UUID, str]) -> Optional[Task]:
        """
        Get a task by ID.
        
        Args:
            task_id: The ID of the task to get
            
        Returns:
            The Task if found, None otherwise
        """
        if isinstance(task_id, str):
            task_id = UUID(task_id)
        return self.tasks.get(task_id)
    
    def save(self, directory: Optional[Path] = None) -> Path:
        """
        Save the project to disk.
        
        Args:
            directory: Directory to save the project to. If None, uses self.directory or creates a new one.
            
        Returns:
            Path to the saved project directory
        """
        if directory is not None:
            self.directory = directory
        
        if self.directory is None:
            # Create a directory based on project name if not specified
            self.directory = Path(f"./projects/{self.name.replace(' ', '_').lower()}")
        
        # Create directory if it doesn't exist
        os.makedirs(self.directory, exist_ok=True)
        
        # Save project metadata
        project_file = self.directory / "project.json"
        with open(project_file, "w") as f:
            # Exclude tasks and directory from serialization
            # Convert UUID to strings for JSON serialization
            project_data = self.model_dump(exclude={"tasks", "directory"}, mode="json")
            json.dump(project_data, f, indent=2)
        
        # Save each task
        tasks_dir = self.directory / "tasks"
        os.makedirs(tasks_dir, exist_ok=True)
        
        for task_id, task in self.tasks.items():
            task_file = tasks_dir / f"{task_id}.json"
            with open(task_file, "w") as f:
                json.dump(task.model_dump(mode="json"), f, indent=2)
        
        return self.directory
    
    @classmethod
    def load(cls, directory: Union[str, Path]) -> "DatasetProject":
        """
        Load a project from disk.
        
        Args:
            directory: Directory to load the project from
            
        Returns:
            Loaded DatasetProject instance
        """
        if isinstance(directory, str):
            directory = Path(directory)
        
        # Load project metadata
        project_file = directory / "project.json"
        with open(project_file, "r") as f:
            project_data = json.load(f)
        
        # Create project instance
        project = cls(**project_data, directory=directory)
        
        # Load tasks
        tasks_dir = directory / "tasks"
        if tasks_dir.exists():
            for task_file in tasks_dir.glob("*.json"):
                with open(task_file, "r") as f:
                    task_data = json.load(f)
                    task = Task(**task_data)
                    project.tasks[task.id] = task
        
        return project 