# Dataset Creator

A tool for generating training datasets to fine-tune smaller language models.

## Overview

This project provides a streamlined workflow for:

1. Preparing question sets or prompts
2. Feeding them to large language models (e.g., GPT-4, Claude)
3. Generating structured training datasets
4. Using these datasets to fine-tune smaller models

## Installation

```bash
# Install from source
git clone https://github.com/yourusername/dataset-creator.git
cd dataset-creator
pip install -e .
```

## Usage

### Basic Usage

```python
from dataset_creator import DatasetProject
from dataset_creator.data_gen import QuestionGenerator
from dataset_creator.fine_tune import ModelFineTuner

# Create a new dataset project
project = DatasetProject(name="my-project")

# Define a task
task = project.create_task(
    name="qa-task",
    instruction="Answer the given question accurately and concisely",
)

# Generate questions
question_gen = QuestionGenerator()
questions = question_gen.generate_from_topics(
    topics=["Science", "History", "Technology"],
    num_questions=30
)

# Feed questions to LLM to create dataset
dataset = task.create_dataset(
    name="training-data",
    model="gpt-4",
    questions=questions
)

# Fine-tune a smaller model
fine_tuner = ModelFineTuner()
fine_tuned_model = fine_tuner.fine_tune(
    dataset=dataset,
    base_model="llama-3-8b",
    epochs=3
)
```

### Command Line Interface

The tool also provides a CLI for easier use:

```bash
# Generate a dataset from a set of topics
dataset-creator generate-dataset --topics "Science,History,Technology" --output my_dataset.jsonl

# Fine-tune a model using a dataset
dataset-creator fine-tune --dataset my_dataset.jsonl --base-model "llama-3-8b"
```

## Features

- Simple, easy-to-use API for dataset generation
- Customizable prompt templates for different LLM providers
- Support for different dataset formats (JSONL, CSV, etc.)
- Integration with popular fine-tuning frameworks

## License

MIT License
