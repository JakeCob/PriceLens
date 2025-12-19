import json
import time
import logging
import threading
from typing import Dict, Any, Optional
from datetime import datetime
import websocket # type: ignore

logger = logging.getLogger(__name__)

class SyncClient:
    """
    Client for syncing detected cards to the Web Dashboard via WebSocket.
    Implements cooldown logic to prevent spamming the same card.
    """
    def __init__(self, url: str = "ws://localhost:3000/api/sync/ingest", auth_token: Optional[str] = None):
        self.url = url
        self.auth_token = auth_token
        self.ws: Optional[websocket.WebSocket] = None
        self.connected = False
        self.last_emission: Dict[str, float] = {}  # Map card_id -> timestamp (seconds)
        self.cooldown_seconds = 300  # 5 minutes
        self.lock = threading.Lock()
        
        # Start connection in background or just lazy? 
        # For simplicity in this demo, strict connection handling is minimized.
        self._connect()

    def _connect(self):
        """Establish WebSocket connection."""
        try:
            self.ws = websocket.create_connection(self.url, timeout=5)
            self.connected = True
            logger.info(f"Connected to Sync Server at {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect to Sync Server: {e}")
            self.connected = False
            self.ws = None

    def emit_card(self, card_data: Dict[str, Any]) -> bool:
        """
        Emit a card_detected event if the card hasn't been validly sent recently.
        Returns True if sent, False if ignored (cooldown) or error.
        
        Args:
            card_data: Dictionary containing card details (must have 'id')
        """
        card_id = card_data.get('id')
        if not card_id:
            logger.warning("Attempted to emit card without ID")
            return False

        with self.lock:
            now = time.time()
            last_time = self.last_emission.get(card_id, 0)
            
            # Check Cooldown
            if now - last_time < self.cooldown_seconds:
                logger.debug(f"Card {card_id} ignored due to cooldown ({int(now - last_time)}s < {self.cooldown_seconds}s)")
                return False
            
            # Format Payload
            payload = self._format_payload(card_data)
            
            # Send
            success = self._send(payload)
            if success:
                self.last_emission[card_id] = now
                return True
            return False

    def _format_payload(self, card_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format the payload strictly according to schema."""
        return {
            "type": "card_detected",
            "data": {
                "card_id": str(card_data.get('id', '')),
                "name": str(card_data.get('name', 'Unknown')),
                "set_name": str(card_data.get('set_name', '')), # normalized key
                "price": float(card_data.get('price', 0.0)),
                "timestamp": datetime.now().isoformat()
            }
        }

    def _send(self, payload: Dict[str, Any]) -> bool:
        """Send JSON payload over WebSocket."""
        if not self.ws or not self.connected:
            self._connect()
        
        if not self.ws:
            return False

        try:
            self.ws.send(json.dumps(payload))
            return True
        except Exception as e:
            logger.error(f"Error sending payload: {e}")
            self.connected = False
            self.ws = None
            return False

    def close(self):
        """Close connection."""
        if self.ws:
            self.ws.close()
            self.connected = False
