"""
SiliconFlow API example for the dataset creator.

This script demonstrates how to use SiliconFlow API for:
1. Generating questions from topics
2. Generating a dataset using SiliconFlow models
3. Fine-tuning a model on SiliconFlow
"""

import os
from pathlib import Path

from dataset_creator import DatasetProject, Dataset
from dataset_creator.data_gen import QuestionGenerator, DataGenerator
from dataset_creator.data_gen.generator import GeneratorConfig
from dataset_creator.data_gen.question_generator import QuestionGeneratorConfig
from dataset_creator.fine_tune import ModelFineTuner
from dataset_creator.fine_tune.fine_tuner import FineTuneConfig, FineTuneProvider


tokens = os.getenv("SiliconflowToken")
# Set your SiliconFlow API key
SILICONFLOW_API_KEY = tokens
# SiliconFlow API URL (default is https://api.siliconflow.cn/v1)
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1"

# 1. Create a project and task
project = DatasetProject(name="qa-project-siliconflow")
task = project.create_task(
    name="general-qa",
    instruction="准确简洁地回答给定问题",
    description="通用问答任务"
)

# Add thinking instruction for chain-of-thought
task.thinking_instruction = (
    "请逐步思考这个问题。考虑不同的方面和解决方法，然后提供你的最终答案。"
)

# Save the project
project_dir = project.save()
print(f"项目保存到: {project_dir}")

# 2. 使用 SiliconFlow 从主题生成问题
# 配置问题生成器
question_config = QuestionGeneratorConfig(
    siliconflow_api_url=SILICONFLOW_API_URL
)

question_gen = QuestionGenerator(
    model="deepseek-ai/DeepSeek-V3",  # SiliconFlow 支持的模型
    provider="siliconflow",
    api_key=SILICONFLOW_API_KEY,
    config=question_config
)

questions = question_gen.generate_from_topics(
    topics=["科学", "历史", "技术"],
    num_questions=15  # 总共生成15个问题
)

print(f"生成了 {len(questions)} 个问题")
for i, q in enumerate(questions[:3]):
    print(f"{i+1}. {q}")

# 3. Generate a dataset using SiliconFlow models
# Configure the data generator to use thinking
gen_config = GeneratorConfig(
    use_thinking=True,
    siliconflow_api_url=SILICONFLOW_API_URL
)

# Create data generator
data_generator = DataGenerator(
    model="deepseek-ai/DeepSeek-R1",  # SiliconFlow 支持的模型
    provider="siliconflow",
    api_key=SILICONFLOW_API_KEY,
    config=gen_config
)

# Generate examples
examples = data_generator.generate_dataset(
    task=task,
    questions=questions
)

# Create dataset
dataset = Dataset(
    name="问答数据集",
    description="通用问答数据集，包含思考链",
    task_id=task.id,
    model_used="deepseek-ai/DeepSeek-R1",
    provider="siliconflow"
)

# Add examples to dataset
for example in examples:
    dataset.add_example(example)

# Create default train split
dataset.create_split("train", list(dataset.examples.keys()))

# Save the dataset
dataset_dir = dataset.save()
print(f"数据集保存到: {dataset_dir}")

# # 4. Fine-tune a model using SiliconFlow
# # Configure fine-tuning
# ft_config = FineTuneConfig(
#     provider=FineTuneProvider.SILICONFLOW,
#     epochs=3,
#     siliconflow_api_url=SILICONFLOW_API_URL
# )

# # Create fine-tuner
# fine_tuner = ModelFineTuner(
#     api_key=SILICONFLOW_API_KEY,
#     config=ft_config
# )

# # Start fine-tuning
# job = fine_tuner.fine_tune(
#     dataset=dataset,
#     base_model="llama-3-8b",  # 假设SiliconFlow支持这个模型
#     output_name="my-qa-model"
# )

# print(f"微调任务已启动，ID: {job.id}")
# print(f"状态: {job.status}")

# if job.error_message:
#     print(f"错误: {job.error_message}") 