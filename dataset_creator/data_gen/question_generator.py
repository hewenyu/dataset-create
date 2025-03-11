"""
Question generator module for creating questions from topics.
"""

from typing import Dict, List, Optional, Union

import openai
from pydantic import BaseModel
from tqdm import tqdm


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
            provider: Provider of the model (e.g., 'openai')
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
        all_questions = []
        
        # Calculate questions per topic if total is specified
        questions_per_topic = self.config.questions_per_topic
        if num_questions:
            questions_per_topic = max(1, num_questions // len(topics))
        
        # Generate questions for each topic
        for topic in tqdm(topics, desc="Generating questions by topic"):
            # First, generate subtopics
            subtopics = self.generate_subtopics(topic, subtopics_per_topic)
            
            # Generate questions for the main topic
            topic_questions = self.generate_questions_for_topic(
                topic, questions_per_topic
            )
            all_questions.extend(topic_questions)
            
            # Generate questions for each subtopic
            for subtopic in subtopics:
                subtopic_questions = self.generate_questions_for_topic(
                    f"{topic} - {subtopic}", 
                    questions_per_topic // subtopics_per_topic
                )
                all_questions.extend(subtopic_questions)
        
        # Limit to requested number if specified
        if num_questions and len(all_questions) > num_questions:
            all_questions = all_questions[:num_questions]
            
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
        else:
            # Default implementation for other providers
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
        else:
            # Default implementation for other providers
            raise NotImplementedError(f"Provider {self.provider} not supported yet") 