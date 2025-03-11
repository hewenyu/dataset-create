"""
Dataset Creator - A tool for generating training datasets to fine-tune LLMs
"""

from dataset_creator.core import DatasetProject, Task, Dataset, DatasetExample, DatasetSplit
from dataset_creator.data_gen import DataGenerator, QuestionGenerator
from dataset_creator.fine_tune import ModelFineTuner

__version__ = "0.1.0"

__all__ = [
    "DatasetProject", 
    "Task", 
    "Dataset", 
    "DatasetExample", 
    "DatasetSplit",
    "DataGenerator", 
    "QuestionGenerator", 
    "ModelFineTuner"
] 