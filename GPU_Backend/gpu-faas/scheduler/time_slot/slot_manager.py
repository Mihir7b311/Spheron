# time_slot/slot_manager.py
from typing import Dict, List, Any
import time

class TimeSlotManager:
    def __init__(self, config: Dict[str, Any]):
        self.min_duration = config["min_duration"]
        self.max_duration = config["max_duration"]
        self.window = config["scheduling_window"]
        self.slots: Dict[str, List[Dict]] = {}  # GPU ID -> list of slots

    async def allocate_slot(self, gpu_id: str, duration: int) -> Dict:
        if duration < self.min_duration or duration > self.max_duration:
            raise ValueError("Invalid duration")

        current_time = time.time()
        new_slot = {
            "start_time": current_time,
            "duration": duration,
            "end_time": current_time + duration
        }
        
        if gpu_id not in self.slots:
            self.slots[gpu_id] = []
        
        self.slots[gpu_id].append(new_slot)
        return new_slot

    async def get_next_available_slot(self, gpu_id: str) -> float:
        if gpu_id not in self.slots or not self.slots[gpu_id]:
            return time.time()
            
        return max(slot["end_time"] for slot in self.slots[gpu_id]) 
