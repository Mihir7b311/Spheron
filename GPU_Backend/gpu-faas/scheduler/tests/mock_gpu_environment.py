# tests/mock_gpu_environment.py
from typing import Dict, List, Optional
import time
import asyncio
from dataclasses import dataclass
import yaml
import random
import uuid
import json

@dataclass
class ModelInfo:
    id: str
    size: int
    load_time: float
    exec_time: float

class MockGPU:
    def __init__(self, gpu_id: str, total_memory: int):
        self.gpu_id = gpu_id
        self.total_memory = total_memory
        self.used_memory = 0
        self.cached_models: Dict[str, ModelInfo] = {}
        self.current_job: Optional[Dict] = None
        self.is_busy = False
        self.execution_queue = asyncio.Queue()
        self.utilization = 0.0

    async def load_model(self, model: ModelInfo) -> bool:
        # Check if model is already cached
        if model.id in self.cached_models:
            return True

        # Check if we have enough memory
        if self.used_memory + model.size > self.total_memory:
            return False

        # Simulate model loading time
        await asyncio.sleep(model.load_time)
        
        self.cached_models[model.id] = model
        self.used_memory += model.size
        return True

    async def execute_job(self, job: Dict) -> Dict:
        self.is_busy = True
        self.current_job = job
        model = job['model']
        
        # Simulate execution time
        await asyncio.sleep(model.exec_time)
        
        result = {
            'job_id': job['id'],
            'status': 'completed',
            'gpu_id': self.gpu_id,
            'execution_time': model.exec_time
        }
        
        self.is_busy = False
        self.current_job = None
        return result

    def evict_model(self, model_id: str) -> bool:
        if model_id in self.cached_models:
            model = self.cached_models[model_id]
            self.used_memory -= model.size
            del self.cached_models[model_id]
            return True
        return False

    def get_state(self) -> Dict:
        return {
            'gpu_id': self.gpu_id,
            'total_memory': self.total_memory,
            'used_memory': self.used_memory,
            'cached_models': list(self.cached_models.keys()),
            'is_busy': self.is_busy,
            'utilization': self.utilization,
            'current_job': self.current_job
        }

class MockGPUEnvironment:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.gpus: Dict[str, MockGPU] = {}
        self.models: Dict[str, ModelInfo] = {}
        self.init_environment()

    def init_environment(self):
        # Initialize GPUs
        num_gpus = self.config['test_environment']['num_gpus']
        gpu_memory = self.config['test_environment']['gpu_memory']
        
        for i in range(num_gpus):
            gpu_id = f'gpu-{i}'
            self.gpus[gpu_id] = MockGPU(gpu_id, gpu_memory)

        # Initialize models
        for model in self.config['test_environment']['models']:
            self.models[model['id']] = ModelInfo(
                id=model['id'],
                size=model['size'],
                load_time=model['load_time'],
                exec_time=model['exec_time']
            )

    async def run_test_scenario(self, scenario_name: str):
        scenario = self.config['test_scenarios'][scenario_name]
        tasks = []
        
        for _ in range(scenario['num_requests']):
            # Create test request based on model distribution
            for model_id, probability in scenario['model_distribution'].items():
                if random.random() < probability:
                    request = {
                        'id': f'job_{uuid.uuid4()}',
                        'model_id': model_id,
                        'timestamp': time.time()
                    }
                    tasks.append(self.process_request(request))

            await asyncio.sleep(scenario['request_interval'])

        return await asyncio.gather(*tasks)

    async def process_request(self, request: Dict) -> Dict:
        # Simulate request processing through scheduler
        model = self.models[request['model_id']]
        
        # Find best GPU based on LALB algorithm
        selected_gpu = await self.find_best_gpu(model)
        
        if not selected_gpu:
            return {
                'request_id': request['id'],
                'status': 'rejected',
                'reason': 'no_available_gpu'
            }

        # Load model if needed
        if model.id not in selected_gpu.cached_models:
            success = await selected_gpu.load_model(model)
            if not success:
                return {
                    'request_id': request['id'],
                    'status': 'rejected',
                    'reason': 'model_load_failed'
                }

        # Execute job
        result = await selected_gpu.execute_job({
            'id': request['id'],
            'model': model
        })

        return result

    async def find_best_gpu(self, model: ModelInfo) -> Optional[MockGPU]:
        # Implement LALB algorithm logic here
        best_gpu = None
        best_score = float('inf')
        
        for gpu in self.gpus.values():
            # Calculate score based on cache presence and load
            cache_score = 0 if model.id in gpu.cached_models else 1
            load_score = gpu.utilization / 100.0
            
            total_score = (
                self.config['scheduler_config']['lalb']['cache_weight'] * cache_score +
                self.config['scheduler_config']['lalb']['load_weight'] * load_score
            )
            
            if total_score < best_score:
                best_score = total_score
                best_gpu = gpu

        return best_gpu

    def get_environment_state(self) -> Dict:
        return {
            'gpus': {gpu_id: gpu.get_state() 
                    for gpu_id, gpu in self.gpus.items()},
            'models': {model_id: vars(model) 
                      for model_id, model in self.models.items()}
        }

# Usage example
async def run_tests():
    env = MockGPUEnvironment('test_config.yaml')
    
    # Run cache hit test scenario
    print("Running cache hit test...")
    cache_results = await env.run_test_scenario('cache_hit_test')
    
    # Run load balance test scenario
    print("Running load balance test...")
    load_results = await env.run_test_scenario('load_balance_test')
    
    # Print environment state
    print("\nFinal environment state:")
    print(json.dumps(env.get_environment_state(), indent=2))

if __name__ == "__main__":
    asyncio.run(run_tests())