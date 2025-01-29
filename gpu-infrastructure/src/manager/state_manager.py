# state_manager.py

from enum import Enum
from typing import Dict, Optional
import time
import logging

class SystemState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class StateManager:
    """Manages infrastructure system state"""
    
    def __init__(self):
        self.current_state = SystemState.INITIALIZING
        self.state_history = []
        self.component_states: Dict[str, str] = {}
        self.last_update = time.time()
        
    def update_state(self, new_state: SystemState, reason: Optional[str] = None) -> None:
        self.state_history.append({
            "state": self.current_state,
            "timestamp": self.last_update,
            "duration": time.time() - self.last_update
        })
        self.current_state = new_state
        self.last_update = time.time()
        logging.info(f"System state changed to {new_state.value}" + 
                    (f": {reason}" if reason else ""))

    def update_component_state(self, component: str, state: str) -> None:
        self.component_states[component] = state
        logging.info(f"Component {component} state updated to {state}")

    def get_system_state(self) -> Dict:
        return {
            "current_state": self.current_state.value,
            "uptime": time.time() - self.state_history[0]["timestamp"] if self.state_history else 0,
            "components": self.component_states.copy(),
            "last_update": self.last_update
        }
    


