# kubernetes_controller/controller.py
from kubernetes import client, config
from typing import Dict, Any
import os

class KubernetesController:
    def __init__(self, k8s_config: Dict[str, Any]):
        self.namespace = k8s_config["namespace"]
        self.service_account = k8s_config["service_account"]
        self.pod_limits = k8s_config["pod_limits"]
        
        # Load kubernetes configuration
        if os.getenv("KUBERNETES_SERVICE_HOST"):
            config.load_incluster_config()
        else:
            config.load_kube_config()
            
        self.v1 = client.CoreV1Api()
        self.custom_objects = client.CustomObjectsApi()

    async def create_pod(self, request: Dict[str, Any], gpu_slice: Dict[str, Any]):
        """Create a Kubernetes pod with GPU resources"""
        
        container = client.V1Container(
            name=request["function_id"],
            image=request["image"],
            resources=client.V1ResourceRequirements(
                limits={
                    "cpu": self.pod_limits["cpu"],
                    "memory": self.pod_limits["memory"],
                    "nvidia.com/gpu": str(gpu_slice["gpu_fraction"])
                }
            ),
            env=[
                client.V1EnvVar(
                    name="NVIDIA_VISIBLE_DEVICES",
                    value=str(gpu_slice["gpu_id"])
                ),
                client.V1EnvVar(
                    name="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
                    value=str(gpu_slice["compute_percentage"])
                )
            ]
        )

        pod_spec = client.V1PodSpec(
            containers=[container],
            service_account_name=self.service_account,
            restart_policy="Never"
        )

        pod_metadata = client.V1ObjectMeta(
            name=f"{request['function_id']}-{gpu_slice['slice_id']}",
            namespace=self.namespace,
            labels={
                "app": "gpu-faas",
                "function": request["function_id"],
                "gpu": str(gpu_slice["gpu_id"])
            }
        )

        pod = client.V1Pod(
            metadata=pod_metadata,
            spec=pod_spec
        )

        return self.v1.create_namespaced_pod(
            namespace=self.namespace,
            body=pod
        )

    async def delete_pod(self, pod_name: str):
        """Delete a Kubernetes pod"""
        return self.v1.delete_namespaced_pod(
            name=pod_name,
            namespace=self.namespace
        )

    async def get_pod_status(self, pod_name: str):
        """Get pod status"""
        return self.v1.read_namespaced_pod_status(
            name=pod_name,
            namespace=self.namespace
        ) 
