from pydantic import BaseModel, Field
from typing import Optional

# Represents GPU-specific requirements for a function deployment
class GPURequirements(BaseModel):
    min_memory: str
    preferred_gpu: Optional[str] = None
    compute_capability: Optional[str] = None

# Represents the function deployment request payload
class FunctionDeployRequest(BaseModel):
    function_id: str = Field(..., description="Unique identifier for the function")
    code: str = Field(..., description="Python code to execute")
    memory: str = Field(..., description="Required memory (e.g., '4Gi')")
    compute_percentage: int = Field(..., ge=0, le=100, description="GPU compute requirement")
    shared: bool = Field(default=False, description="Whether GPU sharing is needed")
    gpu_requirements: GPURequirements

# Represents details about a GPU slice allocated for a function
class GPUSlice(BaseModel):
    gpu_id: str
    slice_id: str
    memory: str
    compute_percentage: int

# Represents the response payload for a deployment request
class DeploymentResponse(BaseModel):
    status: str
    gpu_slice: GPUSlice
    pod_name: str
