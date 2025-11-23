"""
Event Bus System
Decoupled event-driven communication for PriceLens components.
"""

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

class EventBus:
    """
    Simple synchronous event bus.
    Allows components to subscribe to and emit events.
    """
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            cls._instance.handlers = defaultdict(list)
        return cls._instance

    def __init__(self):
        # Singleton init is handled in __new__
        pass

    def subscribe(self, event_name: str, handler: Callable[[Any], None]) -> None:
        """
        Subscribe to an event
        
        Args:
            event_name: Name of event (e.g., 'card.detected')
            handler: Callback function taking data argument
        """
        self.handlers[event_name].append(handler)
        logger.debug(f"Subscribed to {event_name}")

    def emit(self, event_name: str, data: Any = None) -> None:
        """
        Emit an event to all subscribers
        
        Args:
            event_name: Name of event
            data: Event payload
        """
        if event_name in self.handlers:
            for handler in self.handlers[event_name]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Error in handler for {event_name}: {e}")
        
        # Log significant events
        if "frame" not in event_name:  # Reduce noise
            logger.debug(f"Event emitted: {event_name}")

    def clear(self) -> None:
        """Clear all subscriptions"""
        self.handlers.clear()

# Global instance
event_bus = EventBus()
