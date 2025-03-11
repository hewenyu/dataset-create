# Dataset Creator Examples

This directory contains example scripts demonstrating how to use the Dataset Creator library.

## Examples

### Basic Usage

The `basic_usage.py` script demonstrates the complete workflow:

1. Creating a project and task
2. Generating questions from topics
3. Generating a dataset using a large language model
4. Fine-tuning a smaller model on the dataset

To run:

```bash
python examples/basic_usage.py
```

Make sure to set your OpenAI API key in the script or as an environment variable:

```python
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

## CLI Examples

The Dataset Creator also provides a command-line interface. Here are some example commands:

### Create a Project

```bash
dataset-creator create-project --name "qa-project" --description "Question answering project"
```

### Generate Questions

```bash
dataset-creator generate-questions --topics "Science,History,Technology" --num-questions 30 --output-file questions.json
```

### Generate a Dataset

```bash
dataset-creator generate-dataset --task-id "your-task-id" --project-dir "./projects/qa-project" --questions-file questions.json --name "qa-dataset" --model "gpt-4" --use-thinking
```

### Fine-tune a Model

```bash
dataset-creator fine-tune --dataset-dir "./datasets/qa-dataset" --base-model "gpt-3.5-turbo" --output-name "my-qa-model" --epochs 3
``` 