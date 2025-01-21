# tests/integration/test_sharing.py
import pytest
from src.virtual_gpu.manager import VirtualGPUManager
from src.space_sharing.memory_manager import MemoryManager
from src.time_sharing.scheduler import TimeScheduler
from src.common.exceptions import GPUError

@pytest.mark.asyncio
class TestGPUSharing:
    async def test_complete_sharing_flow(self, mock_nvml):
        """Test complete GPU sharing workflow"""
        # Initialize all components
        vgpu_manager = VirtualGPUManager()
        memory_manager = MemoryManager(gpu_id=0, total_memory=8*1024)
        scheduler = TimeScheduler()

        try:
            # 1. Create virtual GPU
            vgpu = await vgpu_manager.create_virtual_gpu(
                memory_mb=1024,
                compute_percentage=25,
                priority=1
            )

            # 2. Create memory partition
            partition = await memory_manager.create_partition(
                size_mb=1024,
                owner_id=vgpu["v_gpu_id"]
            )

            # 3. Register process for scheduling
            process = await scheduler.register_process(
                process_id=vgpu["v_gpu_id"],
                owner_id="test_owner",
                priority=1,
                compute_percentage=25,
                time_quota=1000
            )

            # Verify component integration
            assert vgpu["allocated_memory"] == partition["size_mb"]
            assert vgpu["compute_percentage"] == process["compute_percentage"]

            # 4. Run scheduling cycle
            scheduled_id = await scheduler.schedule_next()
            assert scheduled_id == vgpu["v_gpu_id"]

            # 5. Cleanup
            await vgpu_manager.release_virtual_gpu(vgpu["v_gpu_id"])
            await memory_manager.release_partition(partition["partition_id"])
            await scheduler.unregister_process(process["id"])

        except Exception as e:
            pytest.fail(f"Integration test failed: {str(e)}")

    async def test_multiple_gpu_sharing(self, mock_nvml):
        """Test sharing GPU among multiple processes"""
        vgpu_manager = VirtualGPUManager()
        memory_manager = MemoryManager(gpu_id=0, total_memory=8*1024)
        scheduler = TimeScheduler()

        vgpus = []
        partitions = []
        processes = []

        try:
            # Create multiple virtual GPUs
            for i in range(3):
                # 1. Create virtual GPU
                vgpu = await vgpu_manager.create_virtual_gpu(
                    memory_mb=1024,
                    compute_percentage=25,
                    priority=i
                )
                vgpus.append(vgpu)

                # 2. Create memory partition
                partition = await memory_manager.create_partition(
                    size_mb=1024,
                    owner_id=vgpu["v_gpu_id"]
                )
                partitions.append(partition)

                # 3. Register process
                process = await scheduler.register_process(
                    process_id=vgpu["v_gpu_id"],
                    owner_id=f"owner_{i}",
                    priority=i,
                    compute_percentage=25,
                    time_quota=1000
                )
                processes.append(process)

            # Verify scheduling order (highest priority first)
            scheduled_id = await scheduler.schedule_next()
            assert scheduled_id == vgpus[-1]["v_gpu_id"]  # Highest priority

        finally:
            # Cleanup
            for vgpu, partition, process in zip(vgpus, partitions, processes):
                await vgpu_manager.release_virtual_gpu(vgpu["v_gpu_id"])
                await memory_manager.release_partition(partition["partition_id"])
                await scheduler.unregister_process(process["id"])

    async def test_resource_limits(self, mock_nvml):
        """Test resource limit enforcement"""
        vgpu_manager = VirtualGPUManager()
        memory_manager = MemoryManager(gpu_id=0, total_memory=8*1024)
        scheduler = TimeScheduler()

        # Test memory limit
        with pytest.raises(GPUError):
            await vgpu_manager.create_virtual_gpu(
                memory_mb=10*1024,  # Exceeds total memory
                compute_percentage=25
            )

        # Test compute limit
        with pytest.raises(GPUError):
            await vgpu_manager.create_virtual_gpu(
                memory_mb=1024,
                compute_percentage=150  # Exceeds 100%
            )