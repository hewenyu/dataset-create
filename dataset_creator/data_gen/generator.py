"""
Generator module for creating dataset examples using large language models.
"""

import json
import requests
from typing import Dict, List, Optional, Union, Any

import openai
from pydantic import BaseModel
from tqdm import tqdm

from dataset_creator.core import DatasetExample, Task


class GeneratorConfig(BaseModel):
    """Configuration for data generation"""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    use_thinking: bool = False
    thinking_system_prompt: Optional[str] = None
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
        questions: List[str]
    ) -> List[DatasetExample]:
        """
        Generate a dataset for a task by creating examples for each question.
        
        Args:
            task: Task to generate data for
            questions: List of questions to generate answers for
            
        Returns:
            List of dataset examples
        """
        examples = []
        
        # Format system prompt
        system_prompt = task.format_system_prompt()
        
        # Use tqdm to show progress
        for question in tqdm(questions, desc=f"Generating examples with {self.model}"):
            # Generate example
            example = self.generate_example(
                question=question,
                system_prompt=system_prompt,
                thinking_instruction=task.thinking_instruction if self.config.use_thinking else None
            )
            examples.append(example)
        
        return examples
    
    def generate_example(
        self, 
        question: str,
        system_prompt: str,
        thinking_instruction: Optional[str] = None
    ) -> DatasetExample:
        """
        Generate a single example using the configured model.
        
        Args:
            question: The question to generate an answer for
            system_prompt: System prompt to use
            thinking_instruction: Optional instruction for "thinking" step
            
        Returns:
            A DatasetExample with the generated answer
        """
        # Generate thinking if enabled
        thinking = None
        if self.config.use_thinking and thinking_instruction:
            thinking = self.generate_thinking(question, system_prompt, thinking_instruction)
        
        # Generate answer
        answer = self.generate_answer(question, system_prompt, thinking)
        
        # Create and return example
        return DatasetExample(
            question=question,
            answer=answer,
            system_prompt=system_prompt,
            model_used=self.model,
            thinking=thinking,
            metadata={"provider": self.provider}
        )
    
    def generate_thinking(
        self, 
        question: str, 
        system_prompt: str, 
        thinking_instruction: str
    ) -> str:
        """
        Generate the "thinking" step for a question.
        
        Args:
            question: The question to think about
            system_prompt: Base system prompt
            thinking_instruction: Instruction for thinking
            
        Returns:
            Generated thinking text
        """
        # Create a thinking-specific system prompt
        thinking_system_prompt = self.config.thinking_system_prompt or GeneratorConfig.default_thinking_prompt()
        
        # Format full system prompt with thinking instruction
        full_system_prompt = f"{system_prompt}\n\n{thinking_instruction}\n\n{thinking_system_prompt}"
        
        if self.provider == "openai":
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )
            return response.choices[0].message.content
        elif self.provider == "siliconflow":
            api_url = self.config.siliconflow_api_url or "https://api.siliconflow.cn/v1"
            response = requests.post(
                f"{api_url}/chat/completions",
                headers=self.siliconflow_headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": full_system_prompt},
                        {"role": "user", "content": question}
                    ],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "top_p": self.config.top_p
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            # Default implementation for other providers
            raise NotImplementedError(f"Provider {self.provider} not supported yet")
    
    def generate_answer(
        self, 
        question: str, 
        system_prompt: str,
        thinking: Optional[str] = None
    ) -> str:
        """
        Generate the answer for a question.
        
        Args:
            question: The question to answer
            system_prompt: System prompt to use
            thinking: Optional thinking to include in the context
            
        Returns:
            Generated answer
        """
        if self.provider == "openai":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            
            # Add thinking as assistant message if available
            if thinking:
                messages.append({"role": "assistant", "content": thinking})
                # Add a user message asking for the final answer
                messages.append({"role": "user", "content": "Based on your thinking, what is your final answer?"})
            
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )
            return response.choices[0].message.content
        elif self.provider == "siliconflow":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            
            # Add thinking as assistant message if available
            if thinking:
                messages.append({"role": "assistant", "content": thinking})
                # Add a user message asking for the final answer
                messages.append({"role": "user", "content": "Based on your thinking, what is your final answer?"})
            
            api_url = self.config.siliconflow_api_url or "https://api.siliconflow.cn/v1"
            response = requests.post(
                f"{api_url}/chat/completions",
                headers=self.siliconflow_headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "top_p": self.config.top_p
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            # Default implementation for other providers
            raise NotImplementedError(f"Provider {self.provider} not supported yet") 