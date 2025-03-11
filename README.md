[中文](README_zh.md) | [English](README.md)


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
git clone https://github.com/hewenyu/dataset-creator.git
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

### Multilingual Support

The tool supports generating datasets in different languages. You can specify the language at project, task, or dataset level:

```python
from dataset_creator import DatasetProject
from dataset_creator.core.task import Language
from dataset_creator.data_gen import QuestionGenerator
from dataset_creator.data_gen.generator import GeneratorConfig

# Create a project with Chinese as default language
project = DatasetProject(name="my-chinese-project", language=Language.CHINESE)

# Create a task (will inherit Chinese language from project)
task = project.create_task(
    name="chinese-qa",
    instruction="准确简洁地回答给定问题",
)

# Or explicitly create an English task within a Chinese project
english_task = project.create_task(
    name="english-qa",
    instruction="Answer questions accurately and concisely",
    language=Language.ENGLISH
)

# Configure question generator for Chinese
question_config = QuestionGeneratorConfig(language=Language.CHINESE)
question_gen = QuestionGenerator(config=question_config)

# Generate Chinese questions
questions = question_gen.generate_from_topics(
    topics=["科学", "历史", "技术"],
    num_questions=30
)
```

### Command Line Interface

The tool also provides a CLI for easier use:

```bash
# Generate a dataset from a set of topics
dataset-creator generate-dataset --topics "Science,History,Technology" --output my_dataset.jsonl

# Generate a dataset in Chinese
dataset-creator generate-questions --topics "科学,历史,技术" --language chinese --output chinese_questions.json

# Create a Chinese project
dataset-creator create-project --name "chinese-project" --language chinese

# Create a task with specific language
dataset-creator create-task --project-dir "./projects/my-project" --name "chinese-task" --instruction "用中文回答问题" --language chinese
```
![example](doc/png/example.png)
## Features

- Simple, easy-to-use API for dataset generation
- Customizable prompt templates for different LLM providers
- Support for different dataset formats (JSONL, CSV, etc.)
- Integration with popular fine-tuning frameworks
- Support for multiple model providers including OpenAI and SiliconFlow
- **Multilingual support** for creating datasets in different languages (currently English and Chinese)

## License

MIT License
