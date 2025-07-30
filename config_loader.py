"""
Configuration management module for ML evaluation framework.

This module handles loading and processing of configuration files,
path management, and provides utilities for organizing evaluation outputs.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Manages configuration loading and path generation for evaluation tasks."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Optional custom config directory path
        """
        self.root_path = self._get_project_root()
        self.config_dir = config_dir or os.path.join(self.root_path, "config")
        
        # Load all configurations
        self.paths = self._load_paths_config()
        self.prompt_templates = self._load_prompt_templates()
        self.task_config = self._load_task_config()
    
    def _get_project_root(self) -> str:
        """Get the root path of the project."""
        current_file = Path(__file__).resolve()
        # Navigate up to find project root (adjust levels as needed)
        return str(current_file.parent.parent)
    
    def _load_paths_config(self) -> Dict[str, Any]:
        """Load and process the paths configuration from YAML file."""
        config_path = os.path.join(self.config_dir, "paths.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Paths config not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        base_path = config.get('base_path', self.root_path)
        
        def process_paths(data: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively process paths, replacing placeholders."""
            result = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    result[key] = process_paths(value)
                elif isinstance(value, str):
                    result[key] = value.format(
                        base_path=base_path,
                        project_root=self.root_path
                    )
                else:
                    result[key] = value
            return result
        
        paths = config.get('paths', {})
        return process_paths(paths)
    
    def _load_prompt_templates(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Load prompt templates from YAML files."""
        main_template_path = os.path.join(
            self.config_dir, 
            'prompt_templates.yaml'
        )
        eval_template_path = os.path.join(
            self.config_dir, 
            'prompt_templates_eval.yaml'
        )
        
        main_templates = {}
        eval_templates = {}
        
        if os.path.exists(main_template_path):
            with open(main_template_path, 'r', encoding='utf-8') as f:
                main_templates = yaml.safe_load(f) or {}
        
        if os.path.exists(eval_template_path):
            with open(eval_template_path, 'r', encoding='utf-8') as f:
                eval_templates = yaml.safe_load(f) or {}
        
        return main_templates, eval_templates
    
    def _load_task_config(self) -> Dict[str, Any]:
        """Load task configuration from YAML file."""
        config_path = os.path.join(self.config_dir, 'task_config.yaml')
        
        if not os.path.exists(config_path):
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    @staticmethod
    def get_safe_model_name(model_name: str) -> str:
        """Convert model name to filesystem-safe string."""
        return model_name.replace("/", "_").replace("-", "_").replace(":", "_")
    
    def get_run_base_directory(
        self, 
        prompt_setting: str, 
        others: str, 
        model_name: str,
        root_key: str = "output_root"
    ) -> str:
        """
        Generate base directory for a specific evaluation run.
        
        Args:
            prompt_setting: Prompt configuration (e.g., "all", "no_cot")
            others: Additional settings (e.g., "with_schema", "rag")
            model_name: Model identifier
            root_key: Key in paths config for root directory
            
        Returns:
            Path to run-specific base directory
        """
        if root_key not in self.paths:
            root_key = "output_root"  # Fallback to default
        
        base_output_dir = self.paths.get(root_key, "./outputs")
        safe_model_name = self.get_safe_model_name(model_name)
        
        # Structure: base_output/prompt_setting/others/model_name
        run_dir = os.path.join(
            base_output_dir,
            prompt_setting,
            others or "default",
            safe_model_name
        )
        
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    
    def get_prediction_filepath(
        self, 
        prompt_setting: str, 
        others: str, 
        model_name: str, 
        domain: str,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate filepath for prediction results.
        
        Args:
            prompt_setting: Prompt configuration
            others: Additional settings
            model_name: Model identifier
            domain: Task domain (e.g., "legal", "financial")
            filename: Optional custom filename
            
        Returns:
            Full path to prediction result file
        """
        base_dir = self.get_run_base_directory(prompt_setting, others, model_name)
        filename = filename or f"{domain}_predictions.jsonl"
        return os.path.join(base_dir, filename)
    
    def get_evaluation_filepath(
        self, 
        prompt_setting: str, 
        others: str, 
        model_name: str,
        filename: str = "evaluation_results.json"
    ) -> str:
        """Generate filepath for evaluation results."""
        base_dir = self.get_run_base_directory(prompt_setting, others, model_name)
        return os.path.join(base_dir, filename)
    
    def get_statistics_filepath(
        self, 
        prompt_setting: str, 
        others: str, 
        model_name: str,
        filename: str = "statistics.csv"
    ) -> str:
        """Generate filepath for statistics output."""
        base_dir = self.get_run_base_directory(prompt_setting, others, model_name)
        return os.path.join(base_dir, filename)
    
    def get_intermediate_filepath(
        self, 
        prompt_setting: str, 
        others: str, 
        model_name: str, 
        item_key: str,
        filename: str,
        subdir: str = "intermediate"
    ) -> str:
        """
        Generate filepath for intermediate processing files.
        
        Args:
            prompt_setting: Prompt configuration
            others: Additional settings
            model_name: Model identifier
            item_key: Specific item identifier
            filename: Target filename
            subdir: Subdirectory name for organization
            
        Returns:
            Full path to intermediate file
        """
        base_dir = self.get_run_base_directory(prompt_setting, others, model_name)
        item_dir = os.path.join(base_dir, subdir, item_key)
        os.makedirs(item_dir, exist_ok=True)
        return os.path.join(item_dir, filename)
    
    def get_log_filepath(
        self, 
        prompt_setting: str, 
        others: str, 
        model_name: str,
        log_type: str = "main"
    ) -> str:
        """Generate filepath for log files."""
        base_dir = self.get_run_base_directory(prompt_setting, others, model_name)
        log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, f"{log_type}.log")


# Global configuration instance
_config_manager = None

def get_config_manager(config_dir: Optional[str] = None) -> ConfigManager:
    """Get or create global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir)
    return _config_manager


# Convenience functions for backward compatibility
def load_paths_config() -> Dict[str, Any]:
    """Load paths configuration."""
    return get_config_manager().paths

def load_prompt_templates() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load prompt templates."""
    return get_config_manager().prompt_templates

def load_task_config() -> Dict[str, Any]:
    """Load task configuration."""
    return get_config_manager().task_config

def get_safe_model_name(model_name: str) -> str:
    """Convert model name to filesystem-safe string."""
    return ConfigManager.get_safe_model_name(model_name)


# Initialize global configurations
try:
    PATHS = load_paths_config()
    prompt_templates, prompt_templates_eval = load_prompt_templates()
    task_config = load_task_config()
except Exception as e:
    print(f"Warning: Failed to load configurations: {e}")
    PATHS = {}
    prompt_templates = {}
    prompt_templates_eval = {}
    task_config = {}


# Public API
__all__ = [
    'ConfigManager',
    'get_config_manager',
    'PATHS',
    'prompt_templates',
    'prompt_templates_eval',
    'task_config',
    'load_paths_config',
    'load_prompt_templates',
    'load_task_config',
    'get_safe_model_name',
]