"""Basic tests for PriceLens setup"""

import sys
from pathlib import Path
import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def test_imports():
    """Test that core modules can be imported"""
    try:
        from src.config import Config
        from src.utils.logging_config import setup_logging
        from src.detection.detector_base import DetectorBase
        from src.identification.identifier_base import IdentifierBase
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import module: {e}")


def test_config_loading():
    """Test configuration loading"""
    from src.config import Config

    # Create test config
    test_config = Path(__file__).parent.parent / "config.yaml"
    if test_config.exists():
        config = Config(str(test_config))
        assert config.get('app.name') == "Pokemon Card Price Overlay"
        assert config.get('camera.fps') == 30
        assert config.get('detection.confidence_threshold') == 0.5


def test_directory_structure():
    """Test that required directories exist"""
    root = Path(__file__).parent.parent

    required_dirs = [
        'src',
        'src/detection',
        'src/identification',
        'src/api',
        'src/overlay',
        'src/utils',
        'models',
        'data',
        'data/card_database',
        'data/features',
        'tests',
        'scripts',
        'docker'
    ]

    for dir_name in required_dirs:
        dir_path = root / dir_name
        assert dir_path.exists(), f"Missing directory: {dir_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])