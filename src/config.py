"""Configuration management for PriceLens"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


class Config:
    """Application configuration manager"""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration

        Args:
            config_path: Path to config.yaml file
        """
        # Load environment variables
        load_dotenv()

        # Find config file
        if config_path is None:
            # Look for config.yaml in project root
            root_dir = Path(__file__).parent.parent
            config_path = root_dir / "config.yaml"

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Override with environment variables if present
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # Camera settings
        if os.getenv('CAMERA_SOURCE'):
            self.config['camera']['source'] = int(os.getenv('CAMERA_SOURCE'))
        if os.getenv('CAMERA_WIDTH'):
            self.config['camera']['width'] = int(os.getenv('CAMERA_WIDTH'))
        if os.getenv('CAMERA_HEIGHT'):
            self.config['camera']['height'] = int(os.getenv('CAMERA_HEIGHT'))

        # Performance settings
        if os.getenv('USE_GPU'):
            self.config['performance']['use_gpu'] = os.getenv('USE_GPU').lower() == 'true'

        # API settings
        if os.getenv('API_CACHE_TTL'):
            self.config['api']['cache_ttl'] = int(os.getenv('API_CACHE_TTL'))
        if os.getenv('API_RATE_LIMIT'):
            self.config['api']['rate_limit'] = int(os.getenv('API_RATE_LIMIT'))

        # Debug mode
        if os.getenv('DEBUG'):
            self.config['app']['debug'] = os.getenv('DEBUG').lower() == 'true'

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value

        Args:
            key: Configuration key (can use dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)

    def __repr__(self) -> str:
        return f"Config({self.config})"