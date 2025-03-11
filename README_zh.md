# 数据集创建工具

[中文](README_zh.md) | [English](README.md)

一个用于生成训练数据集以微调小型语言模型的工具。

## 概述

本项目提供了一个流程化的工作流程：

1. 准备问题集或提示
2. 将其馈送给大型语言模型（如GPT-4、Claude等）
3. 生成结构化的训练数据集
4. 使用这些数据集微调小型模型

## 安装

```bash
# 从源代码安装
git clone https://github.com/hewenyu/dataset-creator.git
cd dataset-creator
pip install -e .
```

## 使用方法

### 基本用法

```python
from dataset_creator import DatasetProject
from dataset_creator.data_gen import QuestionGenerator
from dataset_creator.fine_tune import ModelFineTuner

# 创建新的数据集项目
project = DatasetProject(name="my-project")

# 定义一个任务
task = project.create_task(
    name="qa-task",
    instruction="准确简洁地回答给定问题",
)

# 生成问题
question_gen = QuestionGenerator()
questions = question_gen.generate_from_topics(
    topics=["科学", "历史", "技术"],
    num_questions=30
)

# 使用大语言模型创建数据集
dataset = task.create_dataset(
    name="训练数据",
    model="gpt-4",
    questions=questions
)

# 微调小型模型
fine_tuner = ModelFineTuner()
fine_tuned_model = fine_tuner.fine_tune(
    dataset=dataset,
    base_model="llama-3-8b",
    epochs=3
)
```

### 命令行界面

该工具还提供了CLI，便于使用：

```bash
# 从一组主题生成数据集
dataset-creator generate-dataset --topics "科学,历史,技术" --output my_dataset.jsonl

# 使用数据集微调模型
dataset-creator fine-tune --dataset my_dataset.jsonl --base-model "llama-3-8b"
```

## 功能特点

- 简单易用的数据集生成API
- 可定制的不同LLM提供商的提示模板
- 支持不同的数据集格式（JSONL、CSV等）
- 与流行的微调框架集成
- 支持多个模型提供商，包括OpenAI和SiliconFlow

## 许可证

MIT许可证
