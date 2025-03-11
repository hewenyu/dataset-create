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

### Multilingual Support

#### English Example

The `english_example.py` script demonstrates how to use the library with English language settings:

1. Creating a project and task with English as the language
2. Generating English questions from topics
3. Generating a dataset with English question-answer pairs

To run:

```bash
python examples/english_example.py
```

Make sure to set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

#### Chinese Example

The `siliconflow_example.py` script demonstrates how to use the library with Chinese language settings using the SiliconFlow API:

1. Creating a project and task with Chinese as the language
2. Generating Chinese questions from topics
3. Generating a dataset with Chinese question-answer pairs

To run:

```bash
python examples/siliconflow_example.py
```

Make sure to set your SiliconFlow API key:

```bash
export SiliconflowToken="your-siliconflow-api-key-here"
```

### SiliconFlow Usage

The `siliconflow_example.py` script demonstrates how to use SiliconFlow API:

1. Generating questions from topics using SiliconFlow models
2. Generating a dataset using SiliconFlow models
3. Fine-tuning a model on SiliconFlow

To run:

```bash
python examples/siliconflow_example.py
```

Make sure to set your SiliconFlow API key in the script:

```python
SILICONFLOW_API_KEY = "your-siliconflow-api-key-here"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1"  # Default API URL
```

## CLI Examples

The Dataset Creator also provides a command-line interface. Here are some example commands:

### Create a Project

```bash
dataset-creator create-project --name "qa-project" --description "Question answering project"
```

With language specification:
```bash
dataset-creator create-project --name "chinese-qa-project" --language "chinese"
```

### Generate Questions

```bash
dataset-creator generate-questions --topics "Science,History,Technology" --num-questions 30 --output-file questions.json
```

Using SiliconFlow and Chinese language:

```bash
dataset-creator generate-questions --topics "科学,历史,技术" --num-questions 30 --provider siliconflow --api-key "your-api-key" --language chinese --output-file chinese_questions.json
```

### Generate a Dataset

```bash
dataset-creator generate-dataset --task-id "your-task-id" --project-dir "./projects/qa-project" --questions-file questions.json --name "qa-dataset" --model "gpt-4" --use-thinking
```

Using SiliconFlow with Chinese language:

```bash
dataset-creator generate-dataset --task-id "your-task-id" --project-dir "./projects/qa-project" --questions-file chinese_questions.json --name "chinese-qa-dataset" --model "gpt-3.5-turbo" --provider siliconflow --api-key "your-api-key" --language chinese --use-thinking
```

### Fine-tune a Model

```bash
dataset-creator fine-tune --dataset-dir "./datasets/qa-dataset" --base-model "gpt-3.5-turbo" --output-name "my-qa-model" --epochs 3
```

Using SiliconFlow:

```bash
dataset-creator fine-tune --dataset-dir "./datasets/qa-dataset" --base-model "llama-3-8b" --output-name "my-qa-model" --provider siliconflow --api-key "your-api-key" --epochs 3
``` 