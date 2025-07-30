"""
LLM API Interface for AOE Benchmark

This module provides standardized interfaces for calling Large Language Models
for both generation and evaluation tasks in the AOE benchmark.
"""

import re
import time
import logging
from typing import Optional, Tuple, Dict, Any

from openai import OpenAI
from config_loader import PATHS, prompt_templates_eval

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMInterface:
    """Unified interface for LLM API calls with robust error handling."""
    
    # Model context length configurations
    MODEL_LIMITS = {
        # Open Source Models
        "Llama-2-7b-chat-hf": 4096,
        "Llama-3-8B-Instruct": 8192,
        "Llama-3.1-8B-Instruct": 131072,
        "Qwen2.5-72B-Instruct-GPTQ-Int4": 32768,
        "glm-4-9b-chat": 131072,
        
        # Microsoft Models
        "microsoft/phi-4": 16384,
        "microsoft/phi-3.5-mini-128k-instruct": 131072,
        "Phi-4-mini-instruct": 131072,
        "Phi-3-mini": 131072,
        
        # Commercial Models
        "qwen/qwen3-32b": 38000,
        "qwen/qwq-32b": 131072,
        "deepseek/deepseek-r1": 163840,
        "deepseek/deepseek-r1-distill-llama-70b": 131072,
        "google/gemini-2.5-flash-preview": 1048000,
        
        # Other Models
        "Mistral-7B-Instruct-v0.3": 26768,
        "default": 32000,
    }
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize LLM interface.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for API endpoint
        """
        self.api_key = api_key or PATHS.get('api_key', '')
        self.base_url = base_url or PATHS.get('base_url', '')
        
        if not self.api_key:
            logger.warning("No API key provided. Some functionality may be limited.")
    
    def get_model_max_length(self, model_name: str) -> int:
        """Get maximum context length for a given model."""
        return self.MODEL_LIMITS.get(model_name, self.MODEL_LIMITS["default"])
    
    def truncate_prompt(self, prompt: str, model_name: str, max_tokens: int) -> str:
        """
        Truncate prompt to fit within model's context window.
        
        Args:
            prompt: Input prompt text
            model_name: Target model identifier
            max_tokens: Maximum tokens for generation
            
        Returns:
            Truncated prompt that fits within context limits
        """
        model_max_total = self.get_model_max_length(model_name)
        max_prompt_length = model_max_total - (max_tokens * 2)  # Safety buffer
        
        if len(prompt) > max_prompt_length:
            logger.warning(f"Truncating prompt from {len(prompt)} to {max_prompt_length} characters")
            return prompt[:max_prompt_length]
        
        return prompt
    
    def _create_client(self, api_key: str = None, base_url: str = None) -> OpenAI:
        """Create OpenAI client with appropriate configuration."""
        return OpenAI(
            api_key=api_key or self.api_key,
            base_url=base_url or self.base_url
        )
    
    def call_for_evaluation(
        self, 
        query: str, 
        model_name: str = None,
        max_tokens: int = 200,
        max_retries: int = 3,
        **kwargs
    ) -> Optional[str]:
        """
        Call LLM for evaluation tasks.
        
        Args:
            query: Evaluation prompt
            model_name: Model to use (defaults to eval_model_name from config)
            max_tokens: Maximum tokens to generate
            max_retries: Number of retry attempts
            
        Returns:
            Model response or None if failed
        """
        model_name = model_name or PATHS.get('eval_model_name', 'default')
        client = self._create_client(**kwargs)
        
        # Truncate prompt if necessary
        truncated_query = self.truncate_prompt(query, model_name, max_tokens)
        
        logger.info(f"Calling {model_name} for evaluation (max_length: {self.get_model_max_length(model_name)})")
        
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": truncated_query}],
                    max_tokens=max_tokens,
                    temperature=0.1  # Low temperature for consistent evaluation
                )
                
                response = completion.choices[0].message.content
                
                if response:
                    logger.info(f"Evaluation call successful on attempt {attempt + 1}")
                    return response
                else:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.error(f"Evaluation call failed on attempt {attempt + 1}: {e}")
                
            if attempt < max_retries - 1:
                time.sleep(1)  # Brief delay before retry
        
        logger.error(f"Evaluation call failed after {max_retries} attempts")
        return None
    
    def call_for_fractional_score(
        self,
        query: str,
        model_name: str = None,
        max_tokens: int = 20,
        max_retries: int = 3,
        **kwargs
    ) -> Optional[float]:
        """
        Call LLM for fractional scoring tasks.
        
        Expects response in format: <output>0.75</output>
        
        Args:
            query: Scoring prompt
            model_name: Model to use
            max_tokens: Maximum tokens to generate
            max_retries: Number of retry attempts
            
        Returns:
            Score between 0.0 and 1.0, or None if failed
        """
        model_name = model_name or PATHS.get('eval_model_name', 'default')
        client = self._create_client(**kwargs)
        
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": query}],
                    max_tokens=max_tokens,
                    temperature=0.0  # Deterministic scoring
                )
                
                response = completion.choices[0].message.content.strip()
                logger.debug(f"Raw response: {response}")
                
                # Extract score from <output> tags
                match = re.search(r"<output>(.*?)</output>", response, re.IGNORECASE | re.DOTALL)
                
                if match:
                    score_str = match.group(1).strip()
                    try:
                        score = float(score_str)
                        # Clamp score to valid range
                        clamped_score = max(0.0, min(1.0, score))
                        
                        if score != clamped_score:
                            logger.warning(f"Score {score} clamped to {clamped_score}")
                        
                        return clamped_score
                        
                    except ValueError:
                        logger.warning(f"Failed to parse score '{score_str}' from response")
                else:
                    logger.warning(f"No <output> tags found in response: {response}")
                    
            except Exception as e:
                logger.error(f"Fractional scoring call failed on attempt {attempt + 1}: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(1)
        
        logger.error(f"Fractional scoring failed after {max_retries} attempts")
        return None
    
    def call_for_generation(
        self,
        query: str,
        model_name: str,
        max_tokens: int = 1000,
        max_retries: int = 6,
        retry_delay: float = 1.0,
        **kwargs
    ) -> Tuple[Optional[str], bool, int]:
        """
        Call LLM for content generation tasks.
        
        Args:
            query: Generation prompt
            model_name: Model identifier
            max_tokens: Maximum tokens to generate
            max_retries: Number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Tuple of (response, failed, attempts_used)
        """
        client = self._create_client(**kwargs)
        
        # Adjust max_tokens for large context models
        model_max_total = self.get_model_max_length(model_name)
        if model_max_total > 42700:
            max_tokens = 2000
        
        # Truncate prompt if necessary
        truncated_query = self.truncate_prompt(query, model_name, max_tokens)
        
        logger.info(f"Calling {model_name} for generation (max_length: {model_max_total})")
        
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": truncated_query}],
                    max_tokens=max_tokens
                )
                
                response = completion.choices[0].message.content
                
                if response:
                    logger.info(f"Generation call successful on attempt {attempt + 1}")
                    return response, False, attempt + 1
                else:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.error(f"Generation call failed on attempt {attempt + 1}: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        
        logger.error(f"Generation call failed after {max_retries} attempts")
        return None, True, max_retries


# Global instance for backward compatibility
_llm_interface = None

def get_llm_interface(**kwargs) -> LLMInterface:
    """Get or create global LLM interface instance."""
    global _llm_interface
    if _llm_interface is None:
        _llm_interface = LLMInterface(**kwargs)
    return _llm_interface


# Backward compatibility functions
def call_llm_eval(query: str, model_name: str = None, **kwargs) -> Optional[str]:
    """Legacy function for evaluation calls."""
    interface = get_llm_interface()
    return interface.call_for_evaluation(query, model_name, **kwargs)


def call_llm_eval_fractional(query: str, model_name: str = None, **kwargs) -> Optional[float]:
    """Legacy function for fractional scoring calls."""
    interface = get_llm_interface()
    return interface.call_for_fractional_score(query, model_name, **kwargs)


def call_llm_generate(query: str, model_name: str, **kwargs) -> Tuple[Optional[str], bool, int]:
    """Legacy function for generation calls."""
    interface = get_llm_interface()
    return interface.call_for_generation(query, model_name, **kwargs)


def get_model_max_length(model_name: str) -> int:
    """Legacy function for getting model limits."""
    interface = get_llm_interface()
    return interface.get_model_max_length(model_name)