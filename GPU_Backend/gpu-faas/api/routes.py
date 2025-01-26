from fastapi import APIRouter, HTTPException
from .models import FunctionDeployRequest, DeploymentResponse
from scheduler.lalb.scheduler import LALBScheduler
from resource_manager.gpu_slice_manager.manager import GPUSliceManager
from resource_manager.mps_controller.controller import MPSController
from resource_manager.kubernetes_controller.controller import KubernetesController

class GPUFaaSRouter:
    def __init__(
        self,
        scheduler: LALBScheduler,
        gpu_manager: GPUSliceManager,
        mps_controller: MPSController,
        k8s_controller: KubernetesController
    ):
        self.scheduler = scheduler
        self.gpu_manager = gpu_manager
        self.mps_controller = mps_controller
        self.k8s_controller = k8s_controller
        self.router = APIRouter()
        self.setup_routes()

    def setup_routes(self):
        @self.router.post("/function/deploy", response_model=DeploymentResponse)
        async def deploy_function(request: FunctionDeployRequest):
            try:
                # Scheduler Phase
                gpu_assignment = await self.scheduler.schedule_request({
                    "function_id": request.function_id,
                    "memory": request.memory,
                    "compute_percentage": request.compute_percentage,
                    "gpu_requirements": request.gpu_requirements
                })

                # Resource Allocation Phase
                gpu_slice = await self.gpu_manager.allocate_slice(gpu_assignment)

                # MPS Setup Phase
                if request.shared:
                    await self.mps_controller.setup_mps(gpu_slice["gpu_id"])

                # Kubernetes Deployment Phase
                pod = await self.k8s_controller.create_pod({
                    "function_id": request.function_id,
                    "code": request.code,
                    "gpu_slice": gpu_slice
                })

                return DeploymentResponse(
                    status="success",
                    gpu_slice=gpu_slice,
                    pod_name=pod.metadata.name
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
