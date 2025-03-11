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


# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuestionGenerator")


class QuestionGeneratorConfig(BaseModel):
    """Configuration for question generation"""
    temperature: float = 0.8
    max_tokens: int = 2000
    top_p: float = 1.0
    questions_per_topic: int = 10
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
        subtopics_per_topic: int = 3
    ) -> List[str]:
        """
        Generate questions from a list of topics.
        
        Args:
            topics: List of topics to generate questions for
            num_questions: Total number of questions to generate (distributed across topics)
            subtopics_per_topic: Number of subtopics to generate per topic
            
        Returns:
            List of generated questions
        """
        logger.info(f"开始为 {len(topics)} 个主题生成问题，目标总数: {num_questions if num_questions else '未指定'}")
        all_questions = []
        
        # Calculate questions per topic if total is specified
        questions_per_topic = self.config.questions_per_topic
        if num_questions:
            questions_per_topic = max(1, num_questions // len(topics))
        
        logger.info(f"每个主题生成 {questions_per_topic} 个问题，每个主题生成 {subtopics_per_topic} 个子主题")
        
        # Generate questions for each topic
        for topic in tqdm(topics, desc="Generating questions by topic"):
            logger.info(f"处理主题: '{topic}'")
            
            try:
                # First, generate subtopics
                logger.info(f"为主题 '{topic}' 生成子主题")
                start_time = time.time()
                subtopics = self.generate_subtopics(topic, subtopics_per_topic)
                logger.info(f"成功生成 {len(subtopics)} 个子主题，用时 {time.time() - start_time:.2f}秒")
                logger.info(f"子主题列表: {subtopics}")
                
                # Generate questions for the main topic
                logger.info(f"为主题 '{topic}' 生成问题")
                start_time = time.time()
                topic_questions = self.generate_questions_for_topic(
                    topic, questions_per_topic
                )
                logger.info(f"成功为主题 '{topic}' 生成 {len(topic_questions)} 个问题，用时 {time.time() - start_time:.2f}秒")
                all_questions.extend(topic_questions)
                
                # Generate questions for each subtopic
                for subtopic in subtopics:
                    logger.info(f"为子主题 '{subtopic}' 生成问题")
                    start_time = time.time()
                    subtopic_questions = self.generate_questions_for_topic(
                        f"{topic} - {subtopic}", 
                        questions_per_topic // subtopics_per_topic
                    )
                    logger.info(f"成功为子主题 '{subtopic}' 生成 {len(subtopic_questions)} 个问题，用时 {time.time() - start_time:.2f}秒")
                    all_questions.extend(subtopic_questions)
            except Exception as e:
                logger.error(f"处理主题 '{topic}' 时出错: {str(e)}", exc_info=True)
        
        # Limit to requested number if specified
        if num_questions and len(all_questions) > num_questions:
            logger.info(f"限制问题数量从 {len(all_questions)} 到 {num_questions}")
            all_questions = all_questions[:num_questions]
        
        logger.info(f"问题生成完成，总共 {len(all_questions)} 个问题")
        return all_questions
    
    def generate_subtopics(self, topic: str, num_subtopics: int) -> List[str]:
        """
        Generate subtopics for a given topic.
        
        Args:
            topic: The main topic
            num_subtopics: Number of subtopics to generate
            
        Returns:
            List of generated subtopics
        """
        logger.info(f"使用提供商 '{self.provider}'，模型 '{self.model}' 为主题 '{topic}' 生成子主题")
        
        if self.provider == "openai":
            prompt = f"""Generate {num_subtopics} specific subtopics for the topic "{topic}".
            
These subtopics should:
1. Be more specific aspects or areas within the main topic
2. Be diverse and cover different aspects of the main topic
3. Be suitable for generating interesting questions

Return only the list of subtopics, one per line."""
            
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
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
            prompt = f"""Generate {num_subtopics} specific subtopics for the topic "{topic}".
            
These subtopics should:
1. Be more specific aspects or areas within the main topic
2. Be diverse and cover different aspects of the main topic
3. Be suitable for generating interesting questions

Return only the list of subtopics, one per line."""
            
            api_url = self.config.siliconflow_api_url or "https://api.siliconflow.cn/v1"
            endpoint = f"{api_url}/chat/completions"
            
            logger.info(f"调用 SiliconFlow API: {endpoint}")
            logger.info(f"请求模型: {self.model}, 提示内容: '{prompt[:50]}...'")
            
            request_data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
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
                    timeout=120  # 增加超时时间到120秒
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
        else:
            # Default implementation for other providers
            logger.error(f"不支持的提供商: {self.provider}")
            raise NotImplementedError(f"Provider {self.provider} not supported yet")
    
    def generate_questions_for_topic(self, topic: str, num_questions: int) -> List[str]:
        """
        Generate questions for a specific topic.
        
        Args:
            topic: The topic to generate questions for
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        logger.info(f"使用提供商 '{self.provider}'，模型 '{self.model}' 为主题 '{topic}' 生成 {num_questions} 个问题")
        
        if self.provider == "openai":
            prompt = f"""Generate {num_questions} diverse and interesting questions about "{topic}".
            
The questions should:
1. Be clear and specific
2. Vary in difficulty (some easy, some challenging)
3. Cover different aspects of the topic
4. Be suitable for testing knowledge or reasoning about the topic
5. Be phrased as direct questions (not statements)

Return only the list of questions, one per line, without numbering."""
            
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
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
            prompt = f"""Generate {num_questions} diverse and interesting questions about "{topic}".
            
The questions should:
1. Be clear and specific
2. Vary in difficulty (some easy, some challenging)
3. Cover different aspects of the topic
4. Be suitable for testing knowledge or reasoning about the topic
5. Be phrased as direct questions (not statements)

Return only the list of questions, one per line, without numbering."""
            
            api_url = self.config.siliconflow_api_url or "https://api.siliconflow.cn/v1"
            endpoint = f"{api_url}/chat/completions"
            
            logger.info(f"调用 SiliconFlow API: {endpoint}")
            logger.info(f"请求模型: {self.model}, 提示内容: '{prompt[:50]}...'")
            
            request_data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
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
                    timeout=120  # 增加超时时间到120秒
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
        else:
            # Default implementation for other providers
            logger.error(f"不支持的提供商: {self.provider}")
            raise NotImplementedError(f"Provider {self.provider} not supported yet") 