"""
电子女友数据集生成示例

这个脚本演示如何创建用于训练电子女友模型的数据集:
1. 生成与女性伴侣角色相关的问题
2. 使用大型语言模型生成回答和思考链
3. 创建结构化数据集用于微调模型

该示例专注于创建一个温暖、共情且具有女性特质的虚拟伴侣。
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

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("electronic_girlfriend_example.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("electronic_girlfriend_example")

logger.info("开始执行电子女友数据集生成脚本")

# 检查API密钥
# 此示例可以使用SiliconFlow或OpenAI API，取决于可用性
api_key = os.getenv("SiliconflowToken") or os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("环境变量'SiliconflowToken'或'OPENAI_API_KEY'未设置")
    sys.exit("请设置SiliconflowToken或OPENAI_API_KEY环境变量")
logger.info("成功获取API密钥")

# 确定API提供商和模型
if os.getenv("SiliconflowToken"):
    provider = "siliconflow"
    model = "deepseek-ai/DeepSeek-R1"  # SiliconFlow支持的模型
    api_url = "https://api.siliconflow.cn/v1"
else:
    provider = "openai"
    model = "gpt-4"  # OpenAI模型
    api_url = None

logger.info(f"使用API提供商: {provider}, 模型: {model}")

# 1. 创建项目和任务
logger.info("创建电子女友项目和任务")
language = Language.CHINESE  # 设置为中文，也可以改为ENGLISH
project = DatasetProject(name="electronic-girlfriend-project", language=language)
task = project.create_task(
    name="electronic-girlfriend",
    instruction="扮演一个女性角色的电子女友，提供温柔、体贴、共情的对话和情感支持，展现女性特质和魅力",
    description="电子女友对话训练任务，以女性视角回应用户"
)


# 为任务添加思考链指令
task.thinking_instruction = (
    "以女性视角思考问题，考虑女性可能的情感表达和关怀方式。"
    "思考如何表达温柔、体贴的女性特质，并提供情感上的支持和理解。"
    "分析对话情境，从女友角色出发，思考如何让用户感受到亲密感、被关心和被理解。"
    "注意使用温柔亲切的语气，同时保持真诚自然的对话风格。"
)

# 输出思考链指令，确认其设置正确
logger.info(f"设置的思考链指令: {task.thinking_instruction}")

# 保存项目
logger.info("保存项目")
try:
    project_dir = project.save()
    logger.info(f"项目成功保存到: {project_dir}")
    print(f"项目保存到: {project_dir}")
except Exception as e:
    logger.error(f"保存项目时出错: {str(e)}", exc_info=True)
    sys.exit(f"保存项目失败: {str(e)}")

# 2. 生成问题
logger.info("配置问题生成器")
# 配置问题生成器
question_config = QuestionGeneratorConfig(
    temperature=0.8,
    max_tokens=1500,
    questions_per_topic=10,      # 每个主题生成10个问题
    questions_per_subtopic=5,    # 每个子主题生成5个问题
    max_total_questions=200,     # 设置最大总问题数量为200
    language=language
)

logger.info("初始化问题生成器")
try:
    question_gen = QuestionGenerator(
        model=model,
        provider=provider,
        api_key=api_key,
        config=question_config
    )
    logger.info("问题生成器初始化成功")
except Exception as e:
    logger.error(f"初始化问题生成器时出错: {str(e)}", exc_info=True)
    sys.exit(f"初始化问题生成器失败: {str(e)}")

# 定义电子女友相关主题
topics = [
    "日常问候和关心", 
    "情感表达和情侣对话", 
    "女性视角的生活观点",
    "兴趣爱好和休闲活动",
    "浪漫关系和情感支持",
    "关心对方健康和生活",
    "未来规划和共同话题",
    "调情和亲密对话",
    "安慰和情绪支持",
    "日常生活分享和互动"
]
max_total_questions = 200  # 设置最大总问题数量为200
logger.info(f"开始为以下主题生成问题: {topics}, 最大数量: {max_total_questions}")

try:
    questions = question_gen.generate_from_topics(
        topics=topics,
        subtopics_per_topic=4,             # 每个主题生成4个子主题
        questions_per_subtopic=5,          # 每个子主题生成5个问题
        max_total_questions=max_total_questions  # 限制总问题数为200个
    )
    logger.info(f"成功生成 {len(questions)} 个问题")
    for i, q in enumerate(questions[:10]):  # 只记录前10个问题，避免日志过长
        logger.info(f"问题示例 {i+1}: {q}")
    logger.info(f"共生成 {len(questions)} 个问题，日志中只显示前10个示例")
except Exception as e:
    logger.error(f"生成问题时出错: {str(e)}", exc_info=True)
    sys.exit(f"生成问题失败: {str(e)}")

# 3. 生成数据集
logger.info("配置数据生成器")
# 配置数据生成器，确保启用思考链生成
data_config = GeneratorConfig(
    use_thinking=True,  # 启用思考链生成
    temperature=0.8,  # 增加多样性
    max_tokens=2000,
    language=language,
    # 添加自定义的思考系统提示，与task.thinking_instruction配合使用
    thinking_system_prompt=(
        "你是一个以女性视角思考问题的电子女友助手。"
        "请在回答问题前，先仔细思考如何从女友角色出发给予最温暖、体贴的回应。"
        "思考应该包括对用户情感需求的分析，以及如何以女性特质回应这些需求。"
    )
)

logger.info("初始化数据生成器")
try:
    data_generator = DataGenerator(
        model=model,
        provider=provider,
        api_key=api_key,
        config=data_config
    )
    logger.info("数据生成器初始化成功")
    # 显式确认是否启用思考链生成
    logger.info(f"思考链生成启用状态: {data_config.use_thinking}")
    logger.info(f"思考系统提示: {data_config.thinking_system_prompt}")
except Exception as e:
    logger.error(f"初始化数据生成器时出错: {str(e)}", exc_info=True)
    sys.exit(f"初始化数据生成器失败: {str(e)}")

# 设置最大生成示例数
max_examples = 200  # 与问题数量相匹配，确保所有问题都被处理
logger.info(f"设置最大生成示例数量: {max_examples}")

logger.info("开始生成数据集")
try:
    # 确保在生成数据集时传递正确的参数
    logger.info(f"生成数据集时使用的思考链指令: {task.thinking_instruction[:50]}...")
    logger.info(f"思考链启用状态: {data_config.use_thinking}")
    
    examples = data_generator.generate_dataset(
        task=task,
        questions=questions,
        max_examples=max_examples
    )
    logger.info(f"成功生成 {len(examples)} 个示例")
    
    # 验证生成的示例是否包含思考链
    thinking_chains_count = sum(1 for ex in examples if ex.thinking)
    logger.info(f"生成的示例中包含思考链的数量: {thinking_chains_count}/{len(examples)}")
    
    # 如果有示例，输出第一个示例的思考链片段作为验证
    if examples and examples[0].thinking:
        logger.info(f"示例思考链片段: {examples[0].thinking[:100]}...")
    else:
        logger.warning("未在生成的示例中找到思考链，请检查生成配置")
except Exception as e:
    logger.error(f"生成数据集示例时出错: {str(e)}", exc_info=True)
    sys.exit(f"生成数据集示例失败: {str(e)}")

# 创建数据集
logger.info("创建电子女友数据集")
dataset = Dataset(
    name="电子女友对话数据集",
    description="专为女性角色的电子女友设计的对话数据集，包含女性视角的思考链和情感回应",
    task_id=task.id,
    model_used=model,
    provider=provider,
    language=language
)

# 向数据集添加示例
logger.info(f"向数据集添加 {len(examples)} 个示例")
for i, example in enumerate(examples):
    try:
        dataset.add_example(example)
        if i % 5 == 0:  # 每5个示例记录一次日志
            logger.info(f"已添加 {i+1}/{len(examples)} 个示例")
    except Exception as e:
        logger.error(f"添加示例 {i+1} 时出错: {str(e)}")

# 创建训练集划分
logger.info("创建训练集划分")
dataset.create_split("train", list(dataset.examples.keys()))

# 保存数据集
logger.info("保存数据集")
try:
    dataset_dir = dataset.save()
    logger.info(f"电子女友数据集成功保存到: {dataset_dir}")
    print(f"数据集保存到: {dataset_dir}")
except Exception as e:
    logger.error(f"保存数据集时出错: {str(e)}", exc_info=True)
    print(f"保存数据集失败: {str(e)}")

logger.info("电子女友数据集生成脚本执行完成") 