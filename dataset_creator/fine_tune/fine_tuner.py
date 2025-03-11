"""
Fine-tuning module for training smaller models on generated datasets.
"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel

from dataset_creator.core import Dataset


class FineTuneProvider(str, Enum):
    """Supported fine-tuning providers"""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class FineTuneStatus(str, Enum):
    """Status of a fine-tuning job"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FineTuneConfig(BaseModel):
    """Configuration for fine-tuning"""
    provider: FineTuneProvider = FineTuneProvider.OPENAI
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    validation_split: float = 0.1
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    output_dir: Optional[str] = None


class FineTuneJob(BaseModel):
    """Represents a fine-tuning job"""
    id: str
    provider: FineTuneProvider
    base_model: str
    status: FineTuneStatus = FineTuneStatus.PENDING
    created_model: Optional[str] = None
    config: FineTuneConfig
    dataset_path: Path
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = {}


class ModelFineTuner:
    """
    Fine-tuner for training smaller models on generated datasets.
    
    This class handles the fine-tuning process using different providers
    (OpenAI, Hugging Face, or local training).
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[FineTuneConfig] = None
    ):
        """
        Initialize the model fine-tuner.
        
        Args:
            api_key: API key for the provider
            config: Optional fine-tuning configuration
        """
        self.api_key = api_key
        self.config = config or FineTuneConfig()
        
        # Set up provider-specific clients if needed
        if self.config.provider == FineTuneProvider.OPENAI:
            import openai
            if api_key:
                openai.api_key = api_key
    
    def fine_tune(
        self,
        dataset: Dataset,
        base_model: str,
        output_name: Optional[str] = None,
        split_name: str = "train",
        config: Optional[FineTuneConfig] = None
    ) -> FineTuneJob:
        """
        Fine-tune a model on a dataset.
        
        Args:
            dataset: Dataset to fine-tune on
            base_model: Base model to fine-tune
            output_name: Name for the fine-tuned model
            split_name: Name of the split to use for training
            config: Optional fine-tuning configuration
            
        Returns:
            FineTuneJob with information about the fine-tuning job
        """
        # Use provided config or default
        ft_config = config or self.config
        
        # Ensure dataset is saved in the right format
        if not dataset.directory:
            dataset.save()
        
        # Get the path to the formatted dataset file
        dataset_path = self._get_dataset_path(dataset, split_name, ft_config.provider)
        
        # Create a job ID
        job_id = f"ft_{base_model.replace('/', '_')}_{dataset.name}"
        
        # Create the fine-tuning job
        job = FineTuneJob(
            id=job_id,
            provider=ft_config.provider,
            base_model=base_model,
            config=ft_config,
            dataset_path=dataset_path
        )
        
        # Start the fine-tuning job based on provider
        if ft_config.provider == FineTuneProvider.OPENAI:
            self._fine_tune_openai(job, output_name)
        elif ft_config.provider == FineTuneProvider.HUGGINGFACE:
            self._fine_tune_huggingface(job, output_name)
        elif ft_config.provider == FineTuneProvider.LOCAL:
            self._fine_tune_local(job, output_name)
        else:
            raise ValueError(f"Unsupported provider: {ft_config.provider}")
        
        return job
    
    def _get_dataset_path(
        self,
        dataset: Dataset,
        split_name: str,
        provider: FineTuneProvider
    ) -> Path:
        """
        Get the path to the formatted dataset file.
        
        Args:
            dataset: Dataset to get path for
            split_name: Name of the split to use
            provider: Provider to format for
            
        Returns:
            Path to the formatted dataset file
        """
        if not dataset.directory:
            raise ValueError("Dataset must be saved before fine-tuning")
        
        exports_dir = dataset.directory / "exports"
        
        if provider == FineTuneProvider.OPENAI:
            # OpenAI uses chat format
            return exports_dir / f"{split_name}_chat_jsonl.jsonl"
        elif provider == FineTuneProvider.HUGGINGFACE:
            # HuggingFace uses instruction format
            return exports_dir / f"{split_name}_instruction_jsonl.jsonl"
        else:
            # Default to instruction format for local training
            return exports_dir / f"{split_name}_instruction_jsonl.jsonl"
    
    def _fine_tune_openai(self, job: FineTuneJob, output_name: Optional[str] = None) -> None:
        """
        Fine-tune a model using OpenAI's API.
        
        Args:
            job: Fine-tuning job
            output_name: Name for the fine-tuned model
        """
        import openai
        
        try:
            # Create the fine-tuning job
            response = openai.fine_tuning.jobs.create(
                training_file=self._upload_file_openai(job.dataset_path),
                model=job.base_model,
                suffix=output_name,
                hyperparameters={
                    "n_epochs": job.config.epochs,
                    "batch_size": job.config.batch_size,
                    "learning_rate_multiplier": job.config.learning_rate
                }
            )
            
            # Update job with OpenAI job ID
            job.id = response.id
            job.status = FineTuneStatus.RUNNING
            
            print(f"Started OpenAI fine-tuning job: {job.id}")
            print(f"Monitor progress with: openai api fine_tunes.follow -i {job.id}")
            
        except Exception as e:
            job.status = FineTuneStatus.FAILED
            job.error_message = str(e)
            print(f"Error starting OpenAI fine-tuning job: {e}")
    
    def _upload_file_openai(self, file_path: Path) -> str:
        """
        Upload a file to OpenAI for fine-tuning.
        
        Args:
            file_path: Path to the file to upload
            
        Returns:
            OpenAI file ID
        """
        import openai
        
        with open(file_path, "rb") as f:
            response = openai.files.create(
                file=f,
                purpose="fine-tune"
            )
        
        return response.id
    
    def _fine_tune_huggingface(self, job: FineTuneJob, output_name: Optional[str] = None) -> None:
        """
        Fine-tune a model using Hugging Face's transformers.
        
        Args:
            job: Fine-tuning job
            output_name: Name for the fine-tuned model
        """
        # This is a placeholder for the actual implementation
        # In a real implementation, you would use the transformers library
        job.status = FineTuneStatus.PENDING
        job.error_message = "Hugging Face fine-tuning not implemented yet"
        print("Hugging Face fine-tuning not implemented yet")
    
    def _fine_tune_local(self, job: FineTuneJob, output_name: Optional[str] = None) -> None:
        """
        Fine-tune a model locally.
        
        Args:
            job: Fine-tuning job
            output_name: Name for the fine-tuned model
        """
        # This is a placeholder for the actual implementation
        # In a real implementation, you would use a local training script
        job.status = FineTuneStatus.PENDING
        job.error_message = "Local fine-tuning not implemented yet"
        print("Local fine-tuning not implemented yet")
    
    def get_job_status(self, job_id: str) -> FineTuneStatus:
        """
        Get the status of a fine-tuning job.
        
        Args:
            job_id: ID of the job to check
            
        Returns:
            Current status of the job
        """
        # This is a placeholder for the actual implementation
        # In a real implementation, you would check the status with the provider
        return FineTuneStatus.PENDING 