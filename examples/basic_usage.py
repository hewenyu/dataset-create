"""
Basic usage example for the dataset creator.

This script demonstrates how to:
1. Create a project and task
2. Generate questions from topics
3. Generate a dataset using a large language model
4. Fine-tune a smaller model on the dataset
"""

import os
from pathlib import Path

from dataset_creator import DatasetProject
from dataset_creator.data_gen import QuestionGenerator, DataGenerator
from dataset_creator.fine_tune import ModelFineTuner
from dataset_creator.fine_tune.fine_tuner import FineTuneConfig, FineTuneProvider

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# 1. Create a project and task
project = DatasetProject(name="qa-project")
task = project.create_task(
    name="general-qa",
    instruction="Answer the given question accurately and concisely",
    description="General question answering task"
)

# Add thinking instruction for chain-of-thought
task.thinking_instruction = (
    "Think through this question step by step. "
    "Consider different aspects and approaches before providing your final answer."
)

# Save the project
project_dir = project.save()
print(f"Project saved to: {project_dir}")

# 2. Generate questions from topics
question_gen = QuestionGenerator(model="gpt-4")
questions = question_gen.generate_from_topics(
    topics=["Science", "History", "Technology"],
    num_questions=15  # Generate 15 questions total
)

print(f"Generated {len(questions)} questions")
for i, q in enumerate(questions[:3]):
    print(f"{i+1}. {q}")

# 3. Generate a dataset using a large language model
# Configure the data generator to use thinking
from dataset_creator.data_gen.generator import GeneratorConfig
gen_config = GeneratorConfig(use_thinking=True)

# Create data generator
data_generator = DataGenerator(
    model="gpt-4",
    config=gen_config
)

# Generate examples
examples = data_generator.generate_dataset(
    task=task,
    questions=questions
)

# Create dataset
dataset = task.create_dataset(
    name="qa-dataset",
    model="gpt-4",
    questions=questions,
    description="General QA dataset with chain-of-thought"
)

# Save the dataset
dataset_dir = dataset.save()
print(f"Dataset saved to: {dataset_dir}")

# 4. Fine-tune a smaller model on the dataset
# Configure fine-tuning
ft_config = FineTuneConfig(
    provider=FineTuneProvider.OPENAI,
    epochs=3
)

# Create fine-tuner
fine_tuner = ModelFineTuner(config=ft_config)

# Start fine-tuning
job = fine_tuner.fine_tune(
    dataset=dataset,
    base_model="gpt-3.5-turbo",
    output_name="my-qa-model"
)

print(f"Fine-tuning job started with ID: {job.id}")
print(f"Status: {job.status}")

if job.error_message:
    print(f"Error: {job.error_message}") 