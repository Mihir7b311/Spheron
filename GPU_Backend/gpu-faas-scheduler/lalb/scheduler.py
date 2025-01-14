# lalb/scheduler.py
from typing import Dict, List, Any
import time

# lalb/scheduler.py
class LALBScheduler:
    def __init__(self, config: Dict[str, Any], global_queue, local_queue, time_slot):
        self.cache_weight = config["cache_weight"]
        self.load_weight = config["load_weight"]
        self.max_retry = config["max_retry"]
        self.global_queue = global_queue
        self.local_queue = local_queue
        self.time_slot = time_slot
        self.gpu_states = {}  # Track GPU states

    async def get_idle_gpus(self) -> List[str]:
        """Get list of idle GPU IDs"""
        return [
            gpu_id for gpu_id, state in self.gpu_states.items()
            if not state.get('is_busy', False)
        ]

    async def is_model_cached(self, gpu_id: str, model_id: str) -> bool:
        """Check if model is cached on specified GPU"""
        gpu_state = self.gpu_states.get(gpu_id, {})
        return model_id in gpu_state.get('cached_models', set())

    async def dispatch_to_gpu(self, gpu_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch request to specified GPU"""
        await self.local_queue.add_to_gpu_queue(gpu_id, request)
        return {
            "gpu_id": gpu_id,
            "request_id": request.get("function_id"),
            "status": "scheduled"
        }

    async def estimate_finish_time(self, gpu_id: str) -> float:
        """Estimate finish time for GPU including queued requests"""
        queue_length = await self.local_queue.get_gpu_queue_length(gpu_id)
        next_slot = await self.time_slot.get_next_available_slot(gpu_id)
        return next_slot + (queue_length * self.average_execution_time)

    async def schedule_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule a request using LALB algorithm"""
        # First, try to find idle GPUs with cached model
        idle_gpus = await self.get_idle_gpus()
        for gpu_id in idle_gpus:
            if await self.is_model_cached(gpu_id, request["model_id"]):
                return await self.dispatch_to_gpu(gpu_id, request)

        # If no cache hits on idle GPUs, check busy GPUs
        busy_gpus = [gpu_id for gpu_id in self.gpu_states if gpu_id not in idle_gpus]
        for gpu_id in busy_gpus:
            if await self.is_model_cached(gpu_id, request["model_id"]):
                estimated_time = await self.estimate_finish_time(gpu_id)
                if estimated_time < self.estimate_cache_miss_time(request):
                    return await self.dispatch_to_gpu(gpu_id, request)

        # If no suitable GPU found, use least loaded GPU
        target_gpu = await self.find_least_loaded_gpu()
        return await self.dispatch_to_gpu(target_gpu, request)

    async def find_least_loaded_gpu(self) -> str:
        """Find GPU with lowest load"""
        min_load = float('inf')
        target_gpu = None

        for gpu_id, state in self.gpu_states.items():
            load = await self.local_queue.get_gpu_queue_length(gpu_id)
            if load < min_load:
                min_load = load
                target_gpu = gpu_id

        if not target_gpu:
            raise Exception("No available GPUs")
        return target_gpu