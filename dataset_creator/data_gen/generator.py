"""
Generator module for creating dataset examples using large language models.
"""

import json
import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Union, Any

import openai
import requests
from pydantic import BaseModel
from tqdm import tqdm

from dataset_creator.core import DatasetExample, Task
from dataset_creator.core.common import Language

# 配置日志记录
logger = logging.getLogger("DataGenerator")
TimeOut = 360


class GeneratorConfig(BaseModel):
    """Configuration for data generation"""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    use_thinking: bool = False
    thinking_system_prompt: Optional[str] = None
    language: Language = Language.ENGLISH  # 默认为英文
    # SiliconFlow specific configs
    siliconflow_api_url: Optional[str] = "https://api.siliconflow.cn/v1"
    
    @classmethod
    def default_thinking_prompt(cls) -> str:
        """Default system prompt for thinking step"""
        return (
            "Think through the problem step by step. "
            "Be thorough and consider different aspects of the question. "
            "After thinking through the problem, provide a concise answer."
        )


class DataGenerator:
    """
    Generator for creating dataset examples using large language models.
    
    This class handles the interaction with LLM providers to generate
    synthetic data for fine-tuning smaller models.
    """
    
    def __init__(
        self, 
        model: str = "gpt-4", 
        provider: str = "openai",
        api_key: Optional[str] = None,
        config: Optional[GeneratorConfig] = None
    ):
        """
        Initialize the data generator.
        
        Args:
            model: Model to use for generation (e.g., 'gpt-4', 'claude-3-opus')
            provider: Provider of the model (e.g., 'openai', 'anthropic', 'siliconflow')
            api_key: API key for the provider
            config: Optional generation configuration
        """
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.config = config or GeneratorConfig()
        
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
    
    def generate_dataset(
        self, 
        task: Task,
        questions: List[str],
        max_examples: Optional[int] = None
    ) -> List[DatasetExample]:
        """
        Generate a dataset by creating examples for each question.
        
        Args:
            task: The task to generate examples for
            questions: List of questions to generate examples for
            max_examples: Maximum number of examples to generate (None for no limit)
            
        Returns:
            List of generated examples
        """
        logger.info(f"开始为任务 '{task.name}' 生成数据集, 包含 {len(questions)} 个问题, 语言: {task.language.value}")
        examples = []
        
        # 如果指定了最大示例数，确保不超过问题列表长度
        if max_examples is not None:
            logger.info(f"设置生成示例数量上限: {max_examples}")
            actual_questions = questions[:max_examples]
            if len(actual_questions) < len(questions):
                logger.info(f"由于设置了上限 {max_examples}，从 {len(questions)} 个问题中选取了 {len(actual_questions)} 个")
                questions = actual_questions
        
        # Generate system prompt
        system_prompt = task.format_system_prompt()
        logger.info(f"使用系统提示: '{system_prompt[:100]}...'")
        
        # Generate examples for each question
        for question in tqdm(questions, desc="Generating examples"):
            try:
                logger.info(f"处理问题: '{question[:100]}...'")
                start_time = time.time()
                example = self.generate_example(
                    question=question,
                    system_prompt=system_prompt,
                    thinking_instruction=task.thinking_instruction if self.config.use_thinking else None,
                    language=task.language
                )
                logger.info(f"成功生成示例，用时 {time.time() - start_time:.2f}秒")
                examples.append(example)
                
                # 检查是否已达到最大示例数
                if max_examples is not None and len(examples) >= max_examples:
                    logger.info(f"已达到最大示例数量 {max_examples}，停止生成")
                    break
            except Exception as e:
                logger.error(f"为问题 '{question[:50]}...' 生成示例时出错: {str(e)}", exc_info=True)
        
        logger.info(f"数据集生成完成，生成了 {len(examples)}/{len(questions)} 个示例")
        return examples
    
    def generate_example(
        self, 
        question: str,
        system_prompt: str,
        thinking_instruction: Optional[str] = None,
        language: Optional[Language] = None
    ) -> DatasetExample:
        """
        Generate a single example for a question.
        
        Args:
            question: The question to generate an example for
            system_prompt: System prompt to use
            thinking_instruction: Instruction for generating thinking
            language: Language to generate in (default: English)
            
        Returns:
            A DatasetExample
        """
        language = language or self.config.language
        logger.info(f"生成单个示例, 问题: '{question[:50]}...', 语言: {language.value}")
        
        # Generate thinking if enabled
        thinking = None
        answer = None
        if thinking_instruction:
            logger.info("启用了思考链生成")
            start_time = time.time()
            thinking,answer = self.generate_thinking_and_answer(
                question=question,
                system_prompt=system_prompt,
                thinking_instruction=thinking_instruction,
                language=language
            )
            logger.info(f"成功生成思考链，用时 {time.time() - start_time:.2f}秒, 长度: {len(thinking if thinking else '') + len(answer if answer else '')   } 字符")
        
        else:
            # Generate answer
            logger.info("生成回答")
            start_time = time.time()
            answer = self.generate_answer(
                question=question,
                system_prompt=system_prompt,
                language=language
            )
            logger.info(f"成功生成回答，用时 {time.time() - start_time:.2f}秒, 长度: {len(answer)} 字符")
        
        # Create and return the example
        return DatasetExample(
            question=question,
            answer=answer,
            system_prompt=system_prompt,
            thinking=thinking,
            model_used=self.model,
            metadata={
                "provider": self.provider,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "language": language.value
            }
        )
    
    def generate_thinking_and_answer(
        self, 
        question: str, 
        system_prompt: str, 
        thinking_instruction: str,
        language: Optional[Language] = None
    ) -> str:
        """
        Generate thinking for a question.
        
        Args:
            question: The question to generate thinking for
            system_prompt: System prompt to use
            thinking_instruction: Instruction for generating thinking
            language: Language to generate in (default: English)
            
        Returns:
            Generated thinking
        """
        language = language or self.config.language
        logger.info(f"为问题生成思考链, 提供商: {self.provider}, 语言: {language.value}")
        
        # 根据语言添加额外指导
        language_instruction = ""
        if language == Language.CHINESE:
            language_instruction = "请用中文回答。"
        elif language == Language.ENGLISH:
            language_instruction = "Please respond in English."

        if self.provider == "openai":
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"{system_prompt}\n{language_instruction}"},
                    {"role": "user", "content": f"{thinking_instruction}\n\n{question}"}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )
            
            return response.choices[0].message.content
        elif self.provider == "siliconflow":
            try:
                # Combine system prompts
                thinking_system_prompt = (
                    self.config.thinking_system_prompt or 
                    ("你是一个会逐步思考问题的助手。" if language == Language.CHINESE else
                     "You are a thoughtful assistant that thinks through problems step by step.")
                )
                combined_system_prompt = f"{thinking_system_prompt}\n{system_prompt}\n{language_instruction}"
                
                logger.info(f"使用思考系统提示: '{thinking_system_prompt[:50]}...'")
                
                api_url = self.config.siliconflow_api_url or "https://api.siliconflow.cn/v1"
                endpoint = f"{api_url}/chat/completions"
                
                logger.info(f"调用 SiliconFlow API: {endpoint}")
                
                if language == Language.CHINESE:
                    messages = [
                        {"role": "system", "content": combined_system_prompt},
                        {"role": "user", "content": f"{thinking_instruction}\n\n问题: {question}"}
                    ]
                else:  # Default to English
                    messages = [
                        {"role": "system", "content": combined_system_prompt}, 
                        {"role": "user", "content": f"{thinking_instruction}\n\nQuestion: {question}"}
                    ]
                
                request_data = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens or 1000,
                    "top_p": self.config.top_p
                }
                
                logger.info(f"请求数据: {json.dumps(request_data, ensure_ascii=False)[:200]}...")
                
                start_time = time.time()
                response = requests.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=request_data,
                    timeout=TimeOut  # 增加超时时间到120秒
                )
                duration = time.time() - start_time
                logger.info(f"API 响应状态码: {response.status_code}, 用时: {duration:.2f}秒")
                
                if response.status_code != 200:
                    logger.error(f"API调用失败: {response.status_code} - {response.text}")
                    raise Exception(f"SiliconFlow API call failed with status {response.status_code}: {response.text}")
                
                result = response.json()
                if "choices" not in result or len(result["choices"]) == 0:
                    logger.error(f"API响应缺少choices字段: {result}")
                    raise Exception(f"Invalid API response: missing 'choices' field")
                # 思考链和回答
                answer = result["choices"][0]["message"]["content"] 
                thinking = result['choices'][0]['message']['reasoning_content']
                logger.info(f"思考链生成成功，长度: {len(thinking)} 字符")
                return thinking, answer
            except Exception as e:
                logger.error(f"生成思考链时出错: {str(e)}", exc_info=True)
                raise
        else:
            # Default implementation for other providers
            logger.error(f"不支持的提供商: {self.provider}")
            raise NotImplementedError(f"Provider {self.provider} not supported yet")
    
    def generate_answer(
        self, 
        question: str, 
        system_prompt: str,
        thinking: Optional[str] = None,
        language: Optional[Language] = None
    ) -> str:
        """
        Generate answer for a question.
        
        Args:
            question: The question to generate an answer for
            system_prompt: System prompt to use
            thinking: Generated thinking to use (optional)
            language: Language to generate in (default: English)
            
        Returns:
            Generated answer
        """
        language = language or self.config.language
        logger.info(f"为问题生成回答, 提供商: {self.provider}, 语言: {language.value}")
        
        # 根据语言添加额外指导
        language_instruction = ""
        if language == Language.CHINESE:
            language_instruction = "请用中文回答。"
        elif language == Language.ENGLISH:
            language_instruction = "Please respond in English."
        
        if self.provider == "openai":
            messages = [
                {"role": "system", "content": f"{system_prompt}\n{language_instruction}"}
            ]
            
            if thinking:
                # Add thinking as a separate message
                messages.append({"role": "user", "content": f"Question: {question}\n\nThink about this step by step:"})
                messages.append({"role": "assistant", "content": thinking})
                messages.append({"role": "user", "content": "Now provide your final answer:"})
            else:
                # No thinking, just answer the question directly
                messages.append({"role": "user", "content": f"Question: {question}"})
                
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )
            
            return response.choices[0].message.content
        elif self.provider == "siliconflow":
            try:
                messages = [
                    {"role": "system", "content": f"{system_prompt}\n{language_instruction}"}
                ]
                
                if thinking:
                    # Add thinking as a separate message
                    messages.append({"role": "user", "content": f"Question: {question}\n\nThink about this step by step:"})
                    messages.append({"role": "assistant", "content": thinking})
                    messages.append({"role": "user", "content": "Now provide your final answer:"})
                else:
                    # No thinking, just answer the question directly
                    messages.append({"role": "user", "content": f"Question: {question}"})
                
                logger.info(f"消息数量: {len(messages)}")
                
                api_url = self.config.siliconflow_api_url or "https://api.siliconflow.cn/v1"
                endpoint = f"{api_url}/chat/completions"
                
                logger.info(f"调用 SiliconFlow API: {endpoint}")
                
                request_data = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens or 1000,
                    "top_p": self.config.top_p
                }
                
                logger.info(f"请求数据: {json.dumps(request_data, ensure_ascii=False)[:200]}...")
                
                start_time = time.time()
                response = requests.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=request_data,
                    timeout=TimeOut  # 增加超时时间到120秒
                )
                duration = time.time() - start_time
                logger.info(f"API 响应状态码: {response.status_code}, 用时: {duration:.2f}秒")
                
                if response.status_code != 200:
                    logger.error(f"API调用失败: {response.status_code} - {response.text}")
                    raise Exception(f"SiliconFlow API call failed with status {response.status_code}: {response.text}")
                
                result = response.json()
                if "choices" not in result or len(result["choices"]) == 0:
                    logger.error(f"API响应缺少choices字段: {result}")
                    raise Exception(f"Invalid API response: missing 'choices' field")
                
                answer = result["choices"][0]["message"]["content"]
                logger.info(f"回答生成成功，长度: {len(answer)} 字符")
                return answer
            except Exception as e:
                logger.error(f"生成回答时出错: {str(e)}", exc_info=True)
                raise
        else:
            # Default implementation for other providers
            logger.error(f"不支持的提供商: {self.provider}")
            raise NotImplementedError(f"Provider {self.provider} not supported yet") 