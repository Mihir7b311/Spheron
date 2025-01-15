# main.py
from fastapi import FastAPI, HTTPException
from kubernetes_controller.controller import KubernetesController
from mps_controller.controller import MPSController
from gpu_slice_manager.manager import GPUSliceManager
import yaml
import uvicorn

app = FastAPI(title="GPU Resource Manager")

# Load configuration
with open("config/resource_config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize controllers
k8s_controller = KubernetesController(config["kubernetes"])
mps_controller = MPSController(config["mps"])
gpu_slice_manager = GPUSliceManager(config["gpu_slice"])

@app.post("/allocate_resource")
async def allocate_resource(request: dict):
    try:
        # Check available resources
        gpu_slice = await gpu_slice_manager.allocate_slice(request)
        
        # Set up MPS if needed
        if request.get("shared", False):
            await mps_controller.setup_mps(gpu_slice["gpu_id"])
            
        # Create Kubernetes resources
        pod = await k8s_controller.create_pod(request, gpu_slice)
        
        return {
            "status": "success",
            "pod_name": pod.metadata.name,
            "gpu_slice": gpu_slice
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)