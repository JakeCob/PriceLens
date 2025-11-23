"""
Plugin System
Enables extensibility through a plugin architecture.
"""

import logging
import importlib
import pkgutil
from abc import ABC, abstractmethod
from typing import Dict, List, Type

from src.core.event_bus import event_bus

logger = logging.getLogger(__name__)

class Plugin(ABC):
    """Base class for all PriceLens plugins"""
    
    def __init__(self):
        self.enabled = True
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass
        
    @abstractmethod
    def on_load(self) -> None:
        """Called when plugin is loaded"""
        pass
        
    @abstractmethod
    def on_unload(self) -> None:
        """Called when plugin is unloaded"""
        pass

class PluginManager:
    """
    Manages plugin lifecycle and discovery.
    """
    
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        
    def register(self, plugin: Plugin) -> None:
        """Register and load a plugin"""
        if plugin.name in self.plugins:
            logger.warning(f"Plugin {plugin.name} already registered")
            return
            
        try:
            plugin.on_load()
            self.plugins[plugin.name] = plugin
            logger.info(f"Loaded plugin: {plugin.name}")
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin.name}: {e}")

    def unregister(self, plugin_name: str) -> None:
        """Unload and unregister a plugin"""
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            try:
                plugin.on_unload()
            except Exception as e:
                logger.error(f"Error unloading plugin {plugin_name}: {e}")
            del self.plugins[plugin_name]
            logger.info(f"Unloaded plugin: {plugin_name}")

    def load_from_package(self, package_name: str) -> None:
        """
        Discover and load plugins from a python package
        """
        try:
            package = importlib.import_module(package_name)
            
            for _, name, _ in pkgutil.iter_modules(package.__path__):
                full_name = f"{package_name}.{name}"
                try:
                    module = importlib.import_module(full_name)
                    
                    # Find Plugin subclasses
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, Plugin) and 
                            attr is not Plugin):
                            
                            self.register(attr())
                            
                except Exception as e:
                    logger.warning(f"Could not load module {full_name}: {e}")
                    
        except ImportError:
            logger.warning(f"Package {package_name} not found")
