"""
SiliconFlow API example for the dataset creator.

This script demonstrates how to use SiliconFlow API for:
1. Generating questions from topics
2. Generating a dataset using SiliconFlow models
3. Fine-tuning a model on SiliconFlow

This example shows how to create datasets in Chinese language.
"""

import os
import logging
import sys
from pathlib import Path

from dataset_creator import DatasetProject, Dataset
from dataset_creator.core.common import Language
from dataset_creator.data_gen import QuestionGenerator, DataGenerator
from dataset_creator.data_gen.generator import GeneratorConfig
from dataset_creator.data_gen.question_generator import QuestionGeneratorConfig
from dataset_creator.fine_tune import ModelFineTuner
from dataset_creator.fine_tune.fine_tuner import FineTuneConfig, FineTuneProvider

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("siliconflow_example.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("siliconflow_example")

logger.info("开始执行SiliconFlow示例脚本")

# 检查API密钥
tokens = os.getenv("SiliconflowToken")
if not tokens:
    logger.error("环境变量'SiliconflowToken'未设置")
    sys.exit("请设置SiliconflowToken环境变量")
logger.info("成功获取SiliconFlow API密钥")

# Set your SiliconFlow API key
SILICONFLOW_API_KEY = tokens
# SiliconFlow API URL (default is https://api.siliconflow.cn/v1)
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1"
logger.info(f"SiliconFlow API URL: {SILICONFLOW_API_URL}")

# 1. Create a project and task with Chinese language
logger.info("创建中文项目和任务")
project = DatasetProject(name="qa-project-siliconflow", language=Language.CHINESE)
task = project.create_task(
    name="general-qa",
    instruction="准确简洁地回答给定问题",
    description="通用问答任务"
)

# Add thinking instruction for chain-of-thought
task.thinking_instruction = (
    "请逐步思考这个问题。考虑不同的方面和解决方法，然后提供你的最终答案。"
)

# 确认语言设置
logger.info(f"项目语言: {project.language.value}")
logger.info(f"任务语言: {task.language.value}")

# Save the project
logger.info("保存项目")
try:
    project_dir = project.save()
    logger.info(f"项目成功保存到: {project_dir}")
    print(f"项目保存到: {project_dir}")
except Exception as e:
    logger.error(f"保存项目时出错: {str(e)}", exc_info=True)
    sys.exit(f"保存项目失败: {str(e)}")

# 2. 使用 SiliconFlow 从主题生成中文问题
logger.info("配置问题生成器")
# 配置问题生成器
question_config = QuestionGeneratorConfig(
    siliconflow_api_url=SILICONFLOW_API_URL,
    temperature=0.7,  # 调整温度以减少生成的随机性
    max_tokens=1500,  # 减少最大token数以加快响应
    language=Language.CHINESE  # 指定生成中文问题
)

logger.info("初始化问题生成器")
try:
    question_gen = QuestionGenerator(
        model="deepseek-ai/DeepSeek-V3",  # SiliconFlow 支持的模型
        provider="siliconflow",
        api_key=SILICONFLOW_API_KEY,
        config=question_config
    )
    logger.info("问题生成器初始化成功")
except Exception as e:
    logger.error(f"初始化问题生成器时出错: {str(e)}", exc_info=True)
    sys.exit(f"初始化问题生成器失败: {str(e)}")

# 定义要生成问题的主题
topics = ["科学"]
num_questions = 1
logger.info(f"开始为主题 {topics} 生成 {num_questions} 个中文问题")

try:
    # 单独尝试生成第一个主题的子主题
    logger.info(f"尝试为第一个主题 '{topics[0]}' 生成子主题")
    subtopics = question_gen.generate_subtopics(topics[0], 1)
    logger.info(f"成功生成子主题: {subtopics}")

    # 在主题循环中使用较小的子主题数和问题数
    logger.info("现在开始完整的问题生成过程")
    questions = question_gen.generate_from_topics(
        topics=topics,
        num_questions=num_questions,  # 总共生成15个问题
        subtopics_per_topic=1  # 减少子主题数量以加快生成
    )
    
    logger.info(f"成功生成 {len(questions)} 个中文问题")
    print(f"生成了 {len(questions)} 个中文问题")
    for i, q in enumerate(questions[:3]):
        logger.info(f"问题示例 {i+1}: {q}")
        print(f"{i+1}. {q}")
except Exception as e:
    logger.error(f"生成问题时出错: {str(e)}", exc_info=True)
    sys.exit(f"生成问题失败: {str(e)}")

# 3. Generate a dataset using SiliconFlow models in Chinese
logger.info("配置数据生成器")
# Configure data generator
data_config = GeneratorConfig(
    use_thinking=True,  # 启用思考链生成
    temperature=0.7,
    max_tokens=1500,
    language=Language.CHINESE  # 指定中文生成
)

logger.info("初始化数据生成器")
try:
    data_generator = DataGenerator(
        model="deepseek-ai/DeepSeek-R1",  # 使用DeepSeek模型
        provider="siliconflow",
        api_key=SILICONFLOW_API_KEY,
        config=data_config
    )
    logger.info("数据生成器初始化成功")
except Exception as e:
    logger.error(f"初始化数据生成器时出错: {str(e)}", exc_info=True)
    sys.exit(f"初始化数据生成器失败: {str(e)}")

# Define max examples to generate
max_examples = 1  # 限制最大生成数量为1个示例
logger.info(f"设置最大生成示例数量: {max_examples}")

logger.info("开始生成数据集")
try:
    examples = data_generator.generate_dataset(
        task=task,
        questions=questions,
        max_examples=max_examples  # 添加最大示例数量限制
    )
    logger.info(f"成功生成 {len(examples)} 个中文示例")
except Exception as e:
    logger.error(f"生成数据集示例时出错: {str(e)}", exc_info=True)
    sys.exit(f"生成数据集示例失败: {str(e)}")

# Create dataset
logger.info("创建中文数据集")
dataset = Dataset(
    name="中文问答数据集",
    description="通用中文问答数据集，包含思考链",
    task_id=task.id,
    model_used="deepseek-ai/DeepSeek-R1",
    provider="siliconflow"
)

# Add examples to dataset
logger.info(f"向数据集添加 {len(examples)} 个示例")
for i, example in enumerate(examples):
    try:
        dataset.add_example(example)
        if i % 5 == 0:  # 每5个示例记录一次日志
            logger.info(f"已添加 {i+1}/{len(examples)} 个示例")
    except Exception as e:
        logger.error(f"添加示例 {i+1} 时出错: {str(e)}")

# Create default train split
logger.info("创建训练集划分")
dataset.create_split("train", list(dataset.examples.keys()))

# Save the dataset
logger.info("保存数据集")
try:
    dataset_dir = dataset.save()
    logger.info(f"中文数据集成功保存到: {dataset_dir}")
    print(f"中文数据集保存到: {dataset_dir}")
except Exception as e:
    logger.error(f"保存数据集时出错: {str(e)}", exc_info=True)
    print(f"保存数据集失败: {str(e)}")
