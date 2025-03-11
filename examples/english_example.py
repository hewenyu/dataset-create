"""
English dataset generation example for the dataset creator.

This script demonstrates how to use the dataset creator for:
1. Generating questions from topics in English
2. Generating a dataset using OpenAI models in English
3. Fine-tuning a model with English dataset
"""

import os
import logging
import sys
from pathlib import Path

from dataset_creator import DatasetProject, Dataset
from dataset_creator.core.task import Language
from dataset_creator.data_gen import QuestionGenerator, DataGenerator
from dataset_creator.data_gen.generator import GeneratorConfig, Language
from dataset_creator.data_gen.question_generator import QuestionGeneratorConfig
from dataset_creator.fine_tune import ModelFineTuner
from dataset_creator.fine_tune.fine_tuner import FineTuneConfig, FineTuneProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("english_example.log")
    ]
)
logger = logging.getLogger("english_example")

logger.info("Starting English example script")

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("Environment variable 'OPENAI_API_KEY' not set")
    sys.exit("Please set OPENAI_API_KEY environment variable")
logger.info("Successfully retrieved OpenAI API key")

# 1. Create a project and task with English language (default)
logger.info("Creating English project and task")
project = DatasetProject(name="qa-project-english", language=Language.ENGLISH)
task = project.create_task(
    name="general-qa",
    instruction="Answer the given question accurately and concisely",
    description="General question answering task"
)

# Add thinking instruction for chain-of-thought
task.thinking_instruction = (
    "Think through this question step by step. Consider different aspects and approaches, then provide your final answer."
)

# Confirm language settings
logger.info(f"Project language: {project.language.value}")
logger.info(f"Task language: {task.language.value}")

# Save the project
logger.info("Saving project")
try:
    project_dir = project.save()
    logger.info(f"Project successfully saved to: {project_dir}")
    print(f"Project saved to: {project_dir}")
except Exception as e:
    logger.error(f"Error saving project: {str(e)}", exc_info=True)
    sys.exit(f"Failed to save project: {str(e)}")

# 2. Generate English questions from topics
logger.info("Configuring question generator")
# Configure question generator
question_config = QuestionGeneratorConfig(
    temperature=0.7,
    max_tokens=1500,
    language=Language.ENGLISH  # Specify English questions generation
)

logger.info("Initializing question generator")
try:
    question_gen = QuestionGenerator(
        model="gpt-4",  # Use OpenAI model
        provider="openai",
        api_key=api_key,
        config=question_config
    )
    logger.info("Question generator initialized successfully")
except Exception as e:
    logger.error(f"Error initializing question generator: {str(e)}", exc_info=True)
    sys.exit(f"Failed to initialize question generator: {str(e)}")

# Define topics for question generation
topics = ["Science", "History", "Technology"]
num_questions = 15
logger.info(f"Starting to generate {num_questions} English questions for topics {topics}")

try:
    # Try to generate subtopics for the first topic
    logger.info(f"Trying to generate subtopics for the first topic '{topics[0]}'")
    subtopics = question_gen.generate_subtopics(topics[0], 3)
    logger.info(f"Successfully generated subtopics: {subtopics}")

    # Start the complete question generation process
    logger.info("Now starting the complete question generation process")
    questions = question_gen.generate_from_topics(
        topics=topics,
        num_questions=num_questions,
        subtopics_per_topic=2  # Reduce number of subtopics to speed up generation
    )
    
    logger.info(f"Successfully generated {len(questions)} English questions")
    print(f"Generated {len(questions)} English questions")
    for i, q in enumerate(questions[:3]):
        logger.info(f"Question example {i+1}: {q}")
        print(f"{i+1}. {q}")
except Exception as e:
    logger.error(f"Error generating questions: {str(e)}", exc_info=True)
    sys.exit(f"Failed to generate questions: {str(e)}")

# 3. Generate a dataset using OpenAI models in English
logger.info("Configuring data generator")
# Configure the data generator to use thinking
gen_config = GeneratorConfig(
    use_thinking=True,
    temperature=0.7,
    max_tokens=1500,
    language=Language.ENGLISH  # Specify English answers generation
)

# Create data generator
logger.info("Initializing data generator")
try:
    data_generator = DataGenerator(
        model="gpt-4",  # Use OpenAI model
        provider="openai",
        api_key=api_key,
        config=gen_config
    )
    logger.info("Data generator initialized successfully")
except Exception as e:
    logger.error(f"Error initializing data generator: {str(e)}", exc_info=True)
    sys.exit(f"Failed to initialize data generator: {str(e)}")

# Generate examples
logger.info(f"Starting to generate English dataset with {len(questions)} questions")
try:
    examples = data_generator.generate_dataset(
        task=task,
        questions=questions
    )
    logger.info(f"Successfully generated {len(examples)} English examples")
except Exception as e:
    logger.error(f"Error generating dataset examples: {str(e)}", exc_info=True)
    sys.exit(f"Failed to generate dataset examples: {str(e)}")

# Create dataset
logger.info("Creating English dataset")
dataset = Dataset(
    name="English QA Dataset",
    description="General English QA dataset with thinking",
    task_id=task.id,
    model_used="gpt-4",
    provider="openai",
    language=Language.ENGLISH
)

# Add examples to dataset
logger.info(f"Adding {len(examples)} examples to dataset")
for i, example in enumerate(examples):
    try:
        dataset.add_example(example)
        if i % 5 == 0:  # Log every 5 examples
            logger.info(f"Added {i+1}/{len(examples)} examples")
    except Exception as e:
        logger.error(f"Error adding example {i+1}: {str(e)}")

# Create default train split
logger.info("Creating training split")
dataset.create_split("train", list(dataset.examples.keys()))

# Save the dataset
logger.info("Saving dataset")
try:
    dataset_dir = dataset.save()
    logger.info(f"English dataset successfully saved to: {dataset_dir}")
    print(f"English dataset saved to: {dataset_dir}")
except Exception as e:
    logger.error(f"Error saving dataset: {str(e)}", exc_info=True)
    print(f"Failed to save dataset: {str(e)}")

logger.info("English example script completed") 