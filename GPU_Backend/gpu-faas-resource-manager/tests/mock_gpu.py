# tests/mock_gpu.py
class MockGPU:
    def __init__(self, gpu_id, total_memory=16*1024*1024*1024):
        self.gpu_id = gpu_id
        self.total_memory = total_memory
        self.used_memory = 0
        self.processes = []

    def get_memory_info(self):
        return {
            'total': self.total_memory,
            'used': self.used_memory,
            'free': self.total_memory - self.used_memory
        }

# Use in tests
@pytest.fixture
def mock_gpu_environment():
    return {
        'gpu-0': MockGPU('0'),
        'gpu-1': MockGPU('1')
    }