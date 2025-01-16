# gpu-faas-scheduler/main.py
import os
from fastapi import FastAPI, HTTPException
from global_queue.queue_manager import GlobalQueueManager
from local_queue.gpu_queue import LocalQueueManager
from lalb.scheduler import LALBScheduler
from time_slot.slot_manager import TimeSlotManager
import yaml
import uvicorn
from fastapi.responses import JSONResponse
app = FastAPI(title="GPU FaaS Scheduler")



def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config", "scheduler_config.yaml")
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise Exception(f"Configuration file not found at {config_path}")

# Load configuration
config = load_config()

# Initialize components
global_queue = GlobalQueueManager(config["global_queue"])
local_queue = LocalQueueManager(config["local_queue"])
time_slot = TimeSlotManager(config["time_slot"])
scheduler = LALBScheduler(config["lalb"], global_queue, local_queue, time_slot)

@app.post("/schedule")
async def schedule_function(request: dict):
    # return await scheduler.schedule_request(request)
    print("Reponse recived:",request)
    return JSONResponse(content={"message": "Function scheduled"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)