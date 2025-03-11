"""
Command-line interface for the dataset creator.
"""

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from dataset_creator import DatasetProject
from dataset_creator.core import Dataset, Task
from dataset_creator.core.task import Language
from dataset_creator.data_gen import QuestionGenerator
from dataset_creator.fine_tune import ModelFineTuner
from dataset_creator.fine_tune.fine_tuner import FineTuneProvider

app = typer.Typer(help="Dataset Creator CLI")
console = Console()


@app.command()
def create_project(
    name: str = typer.Option(..., help="Name of the project"),
    description: Optional[str] = typer.Option(None, help="Description of the project"),
    output_dir: Optional[str] = typer.Option(None, help="Directory to save the project to"),
    language: str = typer.Option("english", help="Project language (english or chinese)")
):
    """Create a new dataset project."""
    console.print(f"Creating project: [bold]{name}[/bold] with language: [bold]{language}[/bold]")
    
    # Convert language string to Language enum
    lang = Language.ENGLISH
    if language.lower() == "chinese":
        lang = Language.CHINESE
        
    project = DatasetProject(name=name, description=description, language=lang)
    
    if output_dir:
        output_path = project.save(Path(output_dir))
    else:
        output_path = project.save()
    
    console.print(f"Project created and saved to: [bold]{output_path}[/bold]")
    return project


@app.command()
def create_task(
    project_dir: str = typer.Option(..., help="Path to the project directory"),
    name: str = typer.Option(..., help="Name of the task"),
    instruction: str = typer.Option(..., help="Instruction for the task"),
    description: Optional[str] = typer.Option(None, help="Description of the task"),
    thinking_instruction: Optional[str] = typer.Option(None, help="Instruction for thinking step"),
    language: Optional[str] = typer.Option(None, help="Task language (english or chinese, defaults to project language)")
):
    """Create a new task in a project."""
    console.print(f"Loading project from: [bold]{project_dir}[/bold]")
    project = DatasetProject.load(project_dir)
    
    # Convert language string to Language enum if specified
    lang = None
    if language:
        if language.lower() == "chinese":
            lang = Language.CHINESE
        else:
            lang = Language.ENGLISH
    
    console.print(f"Creating task: [bold]{name}[/bold]")
    task = project.create_task(
        name=name,
        instruction=instruction,
        description=description,
        language=lang
    )
    
    if thinking_instruction:
        task.thinking_instruction = thinking_instruction
    
    project.save()
    
    console.print(f"Task created with ID: [bold]{task.id}[/bold]")
    return task


@app.command()
def generate_questions(
    topics: str = typer.Option(..., help="Comma-separated list of topics"),
    num_questions: int = typer.Option(30, help="Number of questions to generate"),
    model: str = typer.Option("gpt-4", help="Model to use for generation"),
    provider: str = typer.Option("openai", help="Provider of the model (openai, siliconflow)"),
    api_key: Optional[str] = typer.Option(None, help="API key for the provider"),
    api_url: Optional[str] = typer.Option(None, help="API URL for the provider (for SiliconFlow)"),
    output_file: str = typer.Option("questions.json", help="File to save questions to"),
    language: str = typer.Option("english", help="Language for questions (english or chinese)")
):
    """Generate questions from topics."""
    # Parse topics
    topic_list = [t.strip() for t in topics.split(",")]
    console.print(f"Generating {num_questions} questions for topics: [bold]{', '.join(topic_list)}[/bold]")
    console.print(f"Using model: [bold]{model}[/bold] from provider: [bold]{provider}[/bold]")
    console.print(f"Language: [bold]{language}[/bold]")
    
    # Set language
    lang = Language.ENGLISH
    if language.lower() == "chinese":
        lang = Language.CHINESE
    
    # Set API key from environment if not provided
    if not api_key:
        if provider == "openai":
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                console.print("[bold red]Error: No OpenAI API key provided. Please set OPENAI_API_KEY environment variable or pass --api-key.[/bold red]")
                return
        elif provider == "siliconflow":
            import os
            api_key = os.getenv("SILICONFLOW_API_KEY")
            if not api_key:
                console.print("[bold red]Error: No SiliconFlow API key provided. Please set SILICONFLOW_API_KEY environment variable or pass --api-key.[/bold red]")
                return
    
    # Configure question generator
    from dataset_creator.data_gen.question_generator import QuestionGeneratorConfig
    config = QuestionGeneratorConfig(
        temperature=0.7,
        language=lang
    )
    
    if api_url and provider == "siliconflow":
        config.siliconflow_api_url = api_url
    
    # Create question generator
    question_gen = QuestionGenerator(
        model=model,
        provider=provider,
        api_key=api_key,
        config=config
    )
    
    # Generate questions
    questions = question_gen.generate_from_topics(
        topics=topic_list,
        num_questions=num_questions
    )
    
    # Save questions to file
    with open(output_file, "w") as f:
        json.dump(questions, f, indent=2)
    
    # Print sample questions
    console.print(f"\nGenerated [bold]{len(questions)}[/bold] questions. Sample:")
    for i, q in enumerate(questions[:5]):
        console.print(f"{i+1}. {q}")
    
    console.print(f"\nQuestions saved to: [bold]{output_file}[/bold]")


@app.command()
def generate_dataset(
    task_id: str = typer.Option(..., help="ID of the task"),
    project_dir: str = typer.Option(..., help="Path to the project directory"),
    questions_file: str = typer.Option(..., help="Path to the questions file"),
    name: str = typer.Option(..., help="Name of the dataset"),
    model: str = typer.Option("gpt-4", help="Model to use for generation"),
    provider: str = typer.Option("openai", help="Provider of the model (openai, siliconflow)"),
    api_key: Optional[str] = typer.Option(None, help="API key for the provider"),
    api_url: Optional[str] = typer.Option(None, help="API URL for the provider (for SiliconFlow)"),
    description: Optional[str] = typer.Option(None, help="Description of the dataset"),
    use_thinking: bool = typer.Option(False, help="Whether to use thinking step"),
    language: Optional[str] = typer.Option(None, help="Language for dataset (english or chinese, defaults to task language)")
):
    """Generate a dataset for a task."""
    console.print(f"Loading project from: [bold]{project_dir}[/bold]")
    project = DatasetProject.load(project_dir)
    
    # Get task
    task = project.get_task(task_id)
    if not task:
        console.print(f"[bold red]Error: Task with ID '{task_id}' not found in project.[/bold red]")
        return
    
    console.print(f"Using task: [bold]{task.name}[/bold]")
    
    # Set language
    lang = None
    if language:
        if language.lower() == "chinese":
            lang = Language.CHINESE
        else:
            lang = Language.ENGLISH
    else:
        lang = task.language
    
    console.print(f"Language for dataset: [bold]{lang.value}[/bold]")
        
    # Load questions
    with open(questions_file, "r") as f:
        questions = json.load(f)
    
    console.print(f"Loaded [bold]{len(questions)}[/bold] questions from [bold]{questions_file}[/bold]")
    
    # Set API key from environment if not provided
    if not api_key:
        if provider == "openai":
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                console.print("[bold red]Error: No OpenAI API key provided. Please set OPENAI_API_KEY environment variable or pass --api-key.[/bold red]")
                return
        elif provider == "siliconflow":
            import os
            api_key = os.getenv("SILICONFLOW_API_KEY")
            if not api_key:
                console.print("[bold red]Error: No SiliconFlow API key provided. Please set SILICONFLOW_API_KEY environment variable or pass --api-key.[/bold red]")
                return
    
    # Configure data generator
    from dataset_creator.data_gen.generator import GeneratorConfig
    config = GeneratorConfig(
        use_thinking=use_thinking,
        temperature=0.7,
        language=lang
    )
    
    if api_url and provider == "siliconflow":
        config.siliconflow_api_url = api_url
    
    # Create data generator
    from dataset_creator.data_gen import DataGenerator
    data_generator = DataGenerator(
        model=model,
        provider=provider,
        api_key=api_key,
        config=config
    )
    
    # Generate examples
    examples = data_generator.generate_dataset(
        task=task,
        questions=questions
    )
    
    # Create dataset
    dataset = Dataset(
        name=name,
        description=description,
        task_id=task.id,
        model_used=model,
        provider=provider
    )
    
    # Add examples to dataset
    for example in examples:
        dataset.add_example(example)
    
    # Create default train split
    dataset.create_split("train", list(dataset.examples.keys()))
    
    # Save dataset
    dataset_dir = dataset.save()
    
    console.print(f"Dataset created with [bold]{len(dataset.examples)}[/bold] examples")
    console.print(f"Dataset saved to: [bold]{dataset_dir}[/bold]")
    return dataset


@app.command()
def fine_tune(
    dataset_dir: str = typer.Option(..., help="Path to the dataset directory"),
    base_model: str = typer.Option(..., help="Base model to fine-tune"),
    output_name: Optional[str] = typer.Option(None, help="Name for the fine-tuned model"),
    provider: str = typer.Option("openai", help="Provider for fine-tuning (openai, siliconflow, huggingface, local)"),
    api_key: Optional[str] = typer.Option(None, help="API key for the provider"),
    api_url: Optional[str] = typer.Option(None, help="API URL for the provider (for SiliconFlow)"),
    epochs: int = typer.Option(3, help="Number of epochs to train for")
):
    """Fine-tune a model on a dataset."""
    console.print(f"Loading dataset from: [bold]{dataset_dir}[/bold]")
    dataset = Dataset.load(dataset_dir)
    
    console.print(f"Fine-tuning [bold]{base_model}[/bold] on dataset [bold]{dataset.name}[/bold]")
    
    from dataset_creator.fine_tune.fine_tuner import FineTuneConfig, FineTuneProvider
    
    # Create fine-tuning config
    config = FineTuneConfig(
        provider=FineTuneProvider(provider),
        epochs=epochs
    )
    if api_url and provider == "siliconflow":
        config.siliconflow_api_url = api_url
    
    # Create fine-tuner
    fine_tuner = ModelFineTuner(api_key=api_key, config=config)
    
    # Start fine-tuning
    job = fine_tuner.fine_tune(
        dataset=dataset,
        base_model=base_model,
        output_name=output_name
    )
    
    console.print(f"Fine-tuning job started with ID: [bold]{job.id}[/bold]")
    console.print(f"Status: [bold]{job.status}[/bold]")
    
    if job.error_message:
        console.print(f"[bold red]Error:[/bold red] {job.error_message}")
    
    return job


if __name__ == "__main__":
    app() 