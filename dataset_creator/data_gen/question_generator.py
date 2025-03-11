"""
Question generator module for creating questions from topics.
"""

import requests
from typing import Dict, List, Optional, Union
import logging
import time

import openai
from pydantic import BaseModel
from tqdm import tqdm

from dataset_creator.core.common import Language


# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuestionGenerator")

TimeOut = 120

class QuestionGeneratorConfig(BaseModel):
    """Configuration for question generation"""
    temperature: float = 0.8
    max_tokens: int = 2000
    top_p: float = 1.0
    questions_per_topic: int = 10
    questions_per_subtopic: int = 3  # 新增：每个子主题的问题数量
    max_total_questions: Optional[int] = None  # 新增：最大总问题数量限制
    language: Language = Language.ENGLISH  # 默认为英文
    system_prompt: str = (
        "You are a helpful assistant that generates diverse, high-quality questions "
        "on specific topics. Your questions should be clear, specific, and varied in "
        "difficulty and style."
    )
    # SiliconFlow specific configs
    siliconflow_api_url: Optional[str] = "https://api.siliconflow.cn/v1"


class QuestionGenerator:
    """
    Generator for creating questions from topics.
    
    This class uses large language models to generate diverse questions
    on specified topics, which can then be used to create training datasets.
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        provider: str = "openai",
        api_key: Optional[str] = None,
        config: Optional[QuestionGeneratorConfig] = None
    ):
        """
        Initialize the question generator.
        
        Args:
            model: Model to use for generation (e.g., 'gpt-4')
            provider: Provider of the model (e.g., 'openai', 'siliconflow')
            api_key: API key for the provider
            config: Optional generation configuration
        """
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.config = config or QuestionGeneratorConfig()
        
        # Set up provider client if needed
        if provider == "openai":
            # Only set key if provided, otherwise use environment
            if api_key:
                openai.api_key = api_key
        elif provider == "siliconflow":
            # SiliconFlow uses an API key in the header
            self.siliconflow_headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
    
    def generate_from_topics(
        self,
        topics: List[str],
        num_questions: Optional[int] = None,
        subtopics_per_topic: int = 3,
        questions_per_subtopic: Optional[int] = None,
        max_total_questions: Optional[int] = None,
        language: Optional[Language] = None
    ) -> List[str]:
        """
        Generate questions from a list of topics.
        
        Args:
            topics: List of topics to generate questions for
            num_questions: 已废弃，请使用 max_total_questions 参数
            subtopics_per_topic: Number of subtopics to generate per topic
            questions_per_subtopic: Number of questions to generate per subtopic
            max_total_questions: Maximum total number of questions to generate
            language: Language to generate questions in (default uses config.language)
            
        Returns:
            List of generated questions
        """
        language = language or self.config.language
        
        # 使用传入的参数或配置中的默认值
        questions_per_subtopic = questions_per_subtopic or self.config.questions_per_subtopic
        questions_per_main_topic = self.config.questions_per_topic
        max_total_questions = max_total_questions or num_questions or self.config.max_total_questions
        
        logger.info(f"开始为 {len(topics)} 个主题生成问题，最大总数: {max_total_questions if max_total_questions else '未指定'}, 语言: {language.value}")
        all_questions = []
        
        # 初始计划的问题总数估算
        questions_per_topic_total = questions_per_main_topic + (subtopics_per_topic * questions_per_subtopic)
        estimated_total = len(topics) * questions_per_topic_total
        
        logger.info(f"计划配置: 每个主题 {questions_per_main_topic} 个问题，每个主题 {subtopics_per_topic} 个子主题，每个子主题 {questions_per_subtopic} 个问题")
        logger.info(f"估计总问题数: {estimated_total}个{' (将被限制为 '+str(max_total_questions)+' 个)' if max_total_questions and estimated_total > max_total_questions else ''}")
        
        # Generate questions for each topic
        for topic_idx, topic in enumerate(tqdm(topics, desc="Generating questions by topic")):
            # 检查是否已达到最大总问题数量
            if max_total_questions and len(all_questions) >= max_total_questions:
                logger.info(f"已达到最大问题数量 {max_total_questions}，停止生成")
                break
                
            logger.info(f"处理主题: '{topic}' ({topic_idx+1}/{len(topics)})")
            
            try:
                # 计算剩余可生成的问题数量
                remaining_questions = max_total_questions - len(all_questions) if max_total_questions else None
                
                # 计算当前主题应生成的问题数量
                current_topic_main_questions = questions_per_main_topic
                if remaining_questions is not None:
                    current_topic_main_questions = min(questions_per_main_topic, remaining_questions)
                    if current_topic_main_questions <= 0:
                        logger.info(f"已达到最大问题数量，跳过主题 '{topic}'")
                        break
                
                # Generate questions for the main topic
                logger.info(f"为主题 '{topic}' 生成 {current_topic_main_questions} 个问题")
                start_time = time.time()
                topic_questions = self.generate_questions_for_topic(
                    topic, current_topic_main_questions, language
                )
                logger.info(f"成功为主题 '{topic}' 生成 {len(topic_questions)} 个问题，用时 {time.time() - start_time:.2f}秒")
                all_questions.extend(topic_questions)
                
                # 再次检查是否已达到最大总问题数量
                if max_total_questions and len(all_questions) >= max_total_questions:
                    logger.info(f"已达到最大问题数量 {max_total_questions}，停止生成子主题")
                    break
                
                # First, generate subtopics
                logger.info(f"为主题 '{topic}' 生成子主题")
                start_time = time.time()
                subtopics = self.generate_subtopics(topic, subtopics_per_topic, language)
                logger.info(f"成功生成 {len(subtopics)} 个子主题，用时 {time.time() - start_time:.2f}秒")
                logger.info(f"子主题列表: {subtopics}")
                
                # Generate questions for each subtopic
                for subtopic_idx, subtopic in enumerate(subtopics):
                    # 计算剩余可生成的问题数量
                    remaining_questions = max_total_questions - len(all_questions) if max_total_questions else None
                    
                    # 如果已达到最大问题数量，退出子主题循环
                    if remaining_questions is not None and remaining_questions <= 0:
                        logger.info(f"已达到最大问题数量，停止生成子主题问题")
                        break
                    
                    # 计算当前子主题应生成的问题数量
                    current_subtopic_questions = questions_per_subtopic
                    if remaining_questions is not None:
                        current_subtopic_questions = min(questions_per_subtopic, remaining_questions)
                        if current_subtopic_questions <= 0:
                            logger.info(f"已达到最大问题数量，跳过子主题 '{subtopic}'")
                            continue
                    
                    logger.info(f"为子主题 '{subtopic}' ({subtopic_idx+1}/{len(subtopics)}) 生成 {current_subtopic_questions} 个问题")
                    start_time = time.time()
                    
                    subtopic_questions = self.generate_questions_for_topic(
                        f"{topic} - {subtopic}", 
                        current_subtopic_questions,
                        language
                    )
                    logger.info(f"成功为子主题 '{subtopic}' 生成 {len(subtopic_questions)} 个问题，用时 {time.time() - start_time:.2f}秒")
                    all_questions.extend(subtopic_questions)
                    
                    # 检查是否已达到最大总问题数量
                    if max_total_questions and len(all_questions) >= max_total_questions:
                        logger.info(f"已达到最大问题数量 {max_total_questions}，停止生成更多子主题问题")
                        break
            except Exception as e:
                logger.error(f"处理主题 '{topic}' 时出错: {str(e)}", exc_info=True)
        
        # 确保不超过最大总问题数量
        if max_total_questions and len(all_questions) > max_total_questions:
            logger.info(f"限制问题数量从 {len(all_questions)} 到 {max_total_questions}")
            all_questions = all_questions[:max_total_questions]
        
        logger.info(f"问题生成完成，总共 {len(all_questions)} 个问题")
        return all_questions
    
    def generate_subtopics(self, topic: str, num_subtopics: int, language: Optional[Language] = None) -> List[str]:
        """
        Generate subtopics for a given topic.
        
        Args:
            topic: The main topic
            num_subtopics: Number of subtopics to generate
            language: Language to generate in (default uses config.language)
            
        Returns:
            List of generated subtopics
        """
        language = language or self.config.language
        logger.info(f"使用提供商 '{self.provider}'，模型 '{self.model}' 为主题 '{topic}' 生成子主题, 语言: {language.value}")
        
        # 根据语言设置提示
        language_instruction = ""
        if language == Language.CHINESE:
            language_instruction = "请用中文生成子主题。"
            prompt = f"""为主题"{topic}"生成{num_subtopics}个具体的子主题。
            
这些子主题应当：
1. 是主题内更具体的方面或领域
2. 多样化，涵盖主题的不同方面
3. 适合用于生成有趣的问题

仅返回子主题列表，每行一个。"""
        else:  # 默认英文
            language_instruction = "Please generate subtopics in English."
            prompt = f"""Generate {num_subtopics} specific subtopics for the topic "{topic}".
            
These subtopics should:
1. Be more specific aspects or areas within the main topic
2. Be diverse and cover different aspects of the main topic
3. Be suitable for generating interesting questions

Return only the list of subtopics, one per line."""
        
        if self.provider == "openai":
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"{self.config.system_prompt}\n{language_instruction}"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )
            
            # Parse the response into a list of subtopics
            subtopics_text = response.choices[0].message.content
            subtopics = [
                line.strip().strip('-').strip() 
                for line in subtopics_text.split('\n') 
                if line.strip()
            ]
            
            # Limit to requested number
            return subtopics[:num_subtopics]
        elif self.provider == "siliconflow":
            try:
                api_url = self.config.siliconflow_api_url or "https://api.siliconflow.cn/v1"
                endpoint = f"{api_url}/chat/completions"
                
                logger.info(f"调用 SiliconFlow API: {endpoint}")
                logger.info(f"请求模型: {self.model}, 提示内容: '{prompt[:50]}...'")
                
                request_data = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": f"{self.config.system_prompt}\n{language_instruction}"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "top_p": self.config.top_p
                }
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        endpoint,
                        headers=self.siliconflow_headers,
                        json=request_data,
                        timeout=TimeOut  # 增加超时时间到120秒
                    )
                    logger.info(f"API 响应状态码: {response.status_code}, 用时: {time.time() - start_time:.2f}秒")
                    
                    if response.status_code != 200:
                        logger.error(f"API 调用失败: {response.status_code} - {response.text}")
                        raise Exception(f"SiliconFlow API call failed with status {response.status_code}: {response.text}")
                    
                    result = response.json()
                    logger.info("成功解析 API 响应")
                    
                    # 记录响应结构
                    if "choices" not in result:
                        logger.error(f"API 响应缺少 'choices' 字段: {result}")
                        raise Exception(f"Invalid API response: missing 'choices' field")
                    
                    # Parse the response into a list of subtopics
                    subtopics_text = result["choices"][0]["message"]["content"]
                    logger.info(f"子主题原始响应: '{subtopics_text[:100]}...'")
                    
                    subtopics = [
                        line.strip().strip('-').strip() 
                        for line in subtopics_text.split('\n') 
                        if line.strip()
                    ]
                    
                    # Limit to requested number
                    subtopics = subtopics[:num_subtopics]
                    logger.info(f"解析出 {len(subtopics)} 个子主题")
                    return subtopics
                except requests.exceptions.Timeout:
                    logger.error(f"SiliconFlow API 请求超时")
                    raise Exception("SiliconFlow API request timed out")
                except requests.exceptions.RequestException as e:
                    logger.error(f"SiliconFlow API 请求错误: {str(e)}")
                    raise Exception(f"SiliconFlow API request failed: {str(e)}")
                except Exception as e:
                    logger.error(f"子主题生成过程中出错: {str(e)}", exc_info=True)
                    raise
            except Exception as e:
                logger.error(f"生成子主题时出错: {str(e)}", exc_info=True)
                raise
        else:
            # Default implementation for other providers
            logger.error(f"不支持的提供商: {self.provider}")
            raise NotImplementedError(f"Provider {self.provider} not supported yet")
    
    def generate_questions_for_topic(self, topic: str, num_questions: int, language: Optional[Language] = None) -> List[str]:
        """
        Generate questions for a specific topic.
        
        Args:
            topic: The topic to generate questions for
            num_questions: Number of questions to generate
            language: Language to generate in (default uses config.language)
            
        Returns:
            List of generated questions
        """
        language = language or self.config.language
        logger.info(f"使用提供商 '{self.provider}'，模型 '{self.model}' 为主题 '{topic}' 生成 {num_questions} 个问题, 语言: {language.value}")
        
        # 根据语言设置提示
        language_instruction = ""
        if language == Language.CHINESE:
            language_instruction = "请用中文生成问题。"
            prompt = f"""为"{topic}"生成{num_questions}个多样化和有趣的问题。
            
这些问题应当：
1. 清晰明确
2. 难度各异（有简单的，也有挑战性的）
3. 涵盖主题的不同方面
4. 适合用于测试知识或推理能力
5. 使用直接的疑问形式（不是陈述句）

仅返回问题列表，每行一个，不要编号。"""
        else:  # 默认英文
            language_instruction = "Please generate questions in English."
            prompt = f"""Generate {num_questions} diverse and interesting questions about "{topic}".
            
The questions should:
1. Be clear and specific
2. Vary in difficulty (some easy, some challenging)
3. Cover different aspects of the topic
4. Be suitable for testing knowledge or reasoning about the topic
5. Be phrased as direct questions (not statements)

Return only the list of questions, one per line, without numbering."""
        
        if self.provider == "openai":
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"{self.config.system_prompt}\n{language_instruction}"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )
            
            # Parse the response into a list of questions
            questions_text = response.choices[0].message.content
            questions = [
                line.strip().strip('-').strip() 
                for line in questions_text.split('\n') 
                if line.strip()
            ]
            
            # Limit to requested number
            return questions[:num_questions]
        elif self.provider == "siliconflow":
            try:
                api_url = self.config.siliconflow_api_url or "https://api.siliconflow.cn/v1"
                endpoint = f"{api_url}/chat/completions"
                
                logger.info(f"调用 SiliconFlow API: {endpoint}")
                logger.info(f"请求模型: {self.model}, 提示内容: '{prompt[:50]}...'")
                
                request_data = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": f"{self.config.system_prompt}\n{language_instruction}"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "top_p": self.config.top_p
                }
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        endpoint,
                        headers=self.siliconflow_headers,
                        json=request_data,
                        timeout=TimeOut  # 增加超时时间到120秒
                    )
                    logger.info(f"API 响应状态码: {response.status_code}, 用时: {time.time() - start_time:.2f}秒")
                    
                    if response.status_code != 200:
                        logger.error(f"API 调用失败: {response.status_code} - {response.text}")
                        raise Exception(f"SiliconFlow API call failed with status {response.status_code}: {response.text}")
                    
                    result = response.json()
                    logger.info("成功解析 API 响应")
                    
                    # 记录响应结构
                    if "choices" not in result:
                        logger.error(f"API 响应缺少 'choices' 字段: {result}")
                        raise Exception(f"Invalid API response: missing 'choices' field")
                    
                    # Parse the response into a list of questions
                    questions_text = result["choices"][0]["message"]["content"]
                    logger.info(f"问题原始响应: '{questions_text[:100]}...'")
                    
                    questions = [
                        line.strip().strip('-').strip() 
                        for line in questions_text.split('\n') 
                        if line.strip()
                    ]
                    
                    # Limit to requested number
                    questions = questions[:num_questions]
                    logger.info(f"解析出 {len(questions)} 个问题")
                    return questions
                except requests.exceptions.Timeout:
                    logger.error(f"SiliconFlow API 请求超时")
                    raise Exception("SiliconFlow API request timed out")
                except requests.exceptions.RequestException as e:
                    logger.error(f"SiliconFlow API 请求错误: {str(e)}")
                    raise Exception(f"SiliconFlow API request failed: {str(e)}")
                except Exception as e:
                    logger.error(f"问题生成过程中出错: {str(e)}", exc_info=True)
                    raise
            except Exception as e:
                logger.error(f"生成问题时出错: {str(e)}", exc_info=True)
                raise
        else:
            # Default implementation for other providers
            logger.error(f"不支持的提供商: {self.provider}")
            raise NotImplementedError(f"Provider {self.provider} not supported yet") 