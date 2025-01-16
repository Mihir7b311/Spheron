# main.py
from fastapi import FastAPI
import yaml
from typing import Dict, Any
from scheduler.lalb.scheduler import LALBScheduler
from scheduler.global_queue.queue_manager import GlobalQueueManager
from scheduler.local_queue.gpu_queue import LocalQueueManager
from scheduler.time_slot.slot_manager import TimeSlotManager
from resource_manager.gpu_slice_manager.manager import GPUSliceManager
from resource_manager.mps_controller.controller import MPSController
from resource_manager.kubernetes_controller.controller import KubernetesController

def init_components(config: Dict[str, Any]):
    # Initialize queue managers first
    global_queue = GlobalQueueManager(config["scheduler"]["global_queue"])
    local_queue = LocalQueueManager(config["scheduler"]["local_queue"])
    time_slot = TimeSlotManager(config["scheduler"]["time_slot"])

    # Initialize scheduler with dependencies
    scheduler = LALBScheduler(
        config=config["scheduler"]["lalb"],
        global_queue=global_queue,
        local_queue=local_queue,
        time_slot=time_slot
    )

    # Initialize other components
    gpu_manager = GPUSliceManager(config["gpu_slice"])
    mps_controller = MPSController(config["mps"])
    k8s_controller = KubernetesController(config["kubernetes"])

    return scheduler, gpu_manager, mps_controller, k8s_controller

def load_config() -> Dict[str, Any]:
    with open("config/integrated_config.yaml", "r") as f:
        return yaml.safe_load(f)

def create_app() -> FastAPI:
    app = FastAPI(title="GPU FaaS Platform")
    
    # Load config and initialize components
    config = load_config()
    scheduler, gpu_manager, mps_controller, k8s_controller = init_components(config)
    
    return app

app = create_app()