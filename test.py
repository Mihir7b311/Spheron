# import re

# def extract_functions_from_code(function_code):
#     # Regular expression to match function definitions
#     pattern = r"def\s+(\w+)\s*\(([^)]*)\):"

#     functions = []

#     # Split the code by lines and check for function definitions
#     lines = function_code.strip().split("\n")

#     for line in lines:
#         match = re.match(pattern, line.strip())

#         if match:
#             funcid = match.group(1)
#             args = match.group(2).strip().split(',') if match.group(2).strip() else []
#             # Clean arguments to remove any unnecessary spaces
#             args = [arg.strip() for arg in args]
#             functions.append({"funcid": funcid, "args": args})

#     return functions

# def write_function_data_to_file(functions, output_file):
#     with open(output_file, "w") as f:
#         for idx, func in enumerate(functions):
#             f.write(f"fn{idx+1}: {func['funcid']}\n")
#             f.write(f"args: {func['args']}\n\n")

# # Example Python code (simulating multiple functions)
# python_code = """
# def example_function(x, y, z):
#     return x + y + z

# def add(a, b):
#     return a + b

# def subtract(m, n):
#     return m - n
# """

# # Extract function details
# functions = extract_functions_from_code(python_code)

# # Write the formatted function data to a file
# write_function_data_to_file(functions, "function_data.txt")


# print("Function data has been written to 'function_data.txt'.")
# def test_function(x):
#     y = x + 1
#     return y
# import pynvml

# pynvml.nvmlInit()
# print("NVML initialized successfully.")
# gpu_count = pynvml.nvmlDeviceGetCount()
# print(f"Detected {gpu_count} GPUs")



# # print(dir(pynvml))
# gpu-management-layer/
# ├── src/
# │   ├── cache_system/             # From gpu-management
# │   │   ├── __init__.py
# │   │   ├── model_cache.py        
# │   │   ├── lru_manager.py       
# │   │   └── memory_tracker.py     
# │   │
# │   ├── execution_engine/         # From execution-engine
# │   │   ├── __init__.py
# │   │   ├── runtime/
# │   │   │   ├── python_executor.py
# │   │   │   └── cuda_context.py 
# │   │   ├── batch_processor/
# │   │   │   ├── batch_manager.py 
# │   │   │   └── inference_batch.py 
# │   │   └── cuda/
# │   │       ├── context_manager.py 
# │   │       └── stream_manager.py 
# │   │
# │   ├── gpu_sharing/             # From gpu-sharing-system  
# │   │   ├── __init__.py
# │   │   ├── virtual_gpu.py
# │   │   ├── time_sharing.py
# │   │   └── space_sharing.py
# │   │
# │   ├── integration/            # New Integration Components
# │   │   ├── __init__.py
# │   │   ├── coordinator.py      # Coordinates between systems
# │   │   ├── resource_manager.py # Unified resource management
# │   │   └── scheduler.py        # Integrated scheduler
# │   │
# │   └── common/                 # Shared Components
#        ├── __init__.py
#        ├── config.py            # Unified configuration
#        ├── monitoring.py        # Combined monitoring
#        ├── metrics.py           # Performance metrics
#        └── exceptions.py        # Common exceptions

# ├── config/
# │   ├── cache_config.yaml       # Cache system config
# │   ├── execution_config.yaml   # Execution engine config  
# │   ├── sharing_config.yaml     # GPU sharing config
# │   └── integrated_config.yaml  # Combined configuration

# └── main.py                   # Main entry point 








# gpu-management-layer/tests/
# ├── conftest.py                     # Global test configuration
# │
# ├── cache_system/                   # Cache System Tests
# │   ├── init.py
# │   ├── test_model_cache.py        # Model cache tests
# │   ├── test_lru_manager.py        # LRU manager tests
# │   └── test_memory_tracker.py     # Memory tracker tests
# │
# ├── execution_engine/              # Execution Engine Tests
# │   ├── init.py
# │   ├── runtime/
# │   │   ├── test_python_executor.py
# │   │   └── test_cuda_context.py
# │   ├── batch_processor/
# │   │   ├── test_batch_manager.py
# │   │   └── test_inference_batch.py
# │   └── cuda/
# │       ├── test_context_manager.py
# │       └── test_stream_manager.py
# │
# ├── gpu_sharing/                  # GPU Sharing Tests
# │   ├── init.py
# │   ├── test_virtual_gpu.py
# │   ├── test_time_sharing.py
# │   └── test_space_sharing.py
# │
# ├── integration/                  # Integration Tests
# │   ├── init.py
# │   ├── test_coordinator.py
# │   ├── test_resource_manager.py
# │   └── test_scheduler.py
# │
# └── common/                      # Common Component Tests
#     ├── init.py
#     ├── test_config.py
#     ├── test_monitoring.py
#     ├── test_metrics.py
#     └── test_exceptions.py






# gpu-faas/
# ├── scheduler/                    # Scheduler Service
# │   ├── global_queue/
# │   │   ├── __init__.py
# │   │   └── queue_manager.py
# │   ├── local_queue/
# │   │   ├── __init__.py
# │   │   └── gpu_queue.py
# │   ├── lalb/
# │   │   ├── __init__.py
# │   │   └── scheduler.py
# │   └── time_slot/
# │       ├── __init__.py
# │       └── slot_manager.py

# ├── resource_manager/            # Resource Manager
# │   ├── gpu_slice_manager/
# │   │   ├── __init__.py
# │   │   └── manager.py
# │   ├── mps_controller/
# │   │   ├── __init__.py
# │   │   └── controller.py
# │   └── kubernetes_controller/
# │       ├── __init__.py
# │       └── controller.py

# ├── api/                        # API Layer
# │   ├── __init__.py
# │   ├── routes.py
# │   └── models.py

# ├── config/                     # Configuration
# │   ├── integrated_config.yaml
# │   ├── scheduler_config.yaml
# │   └── resource_config.yaml

# ├── tests/                      # Test Suite
# │   ├── __init__.py
# │   ├── test_global_queue.py
# │   ├── test_integration.py
# │   ├── test_lalb.py
# │   └── test_local_queue.py

# └── main.py                     # Main Application




gpu-infrastructure/
├── src/
│   ├── __init__.py
│   ├── base/                      # Base Components
│   │   ├── __init__.py
│   │   ├── mps_daemon.py         # MPS Daemon implementation
│   │   ├── gpu_slice.py          # GPU Slice management
│   │   └── exceptions.py         # Custom exceptions
│   │
│   ├── manager/                  # Infrastructure Management
│   │   ├── __init__.py
│   │   ├── infrastructure.py     # Main infrastructure manager
│   │   ├── resource_tracker.py   # Resource tracking
│   │   └── state_manager.py      # State management
│   │
│   ├── hardware/                 # Hardware Interface
│   │   ├── __init__.py
│   │   ├── gpu_device.py        # GPU device interface
│   │   ├── memory_manager.py    # Memory management
│   │   └── compute_manager.py   # Compute resource management
│   │
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── monitoring.py        # Resource monitoring
│       ├── validation.py        # Resource validation
│       └── logging.py          # Logging utilities
│
├── config/                      # Configuration
│   ├── infrastructure_config.yaml  # Main configuration
│   ├── mps_config.yaml            # MPS daemon configuration
│   └── monitoring_config.yaml      # Monitoring configuration




storage-layer/
├── src/
│   ├── redis/                 # Redis Components
│   │   ├── __init__.py
│   │   ├── client.py         # Redis client implementation
│   │   ├── cache.py         # Caching logic
│   │   └── queues.py        # Queue management
│   │
│   ├── postgresql/          # PostgreSQL Components
│   │   ├── __init__.py 
│   │   ├── client.py        # PostgreSQL client
│   │   ├── models.py        # Database models
│   │   ├── migrations/      # Database migrations
│   │   │   ├── __init__.py
│   │   │   └── versions/
│   │   └── repositories/    # Data access layers
│   │       ├── __init__.py
│   │       ├── model_repo.py
│   │       └── metrics_repo.py
│   │
│   ├── model_store/         # Model Storage
│   │   ├── __init__.py
│   │   ├── store.py         # Model storage implementation
│   │   ├── mapper.py        # Memory mapping logic
│   │   └── versioning.py    # Model versioning
│   │
│   ├── common/             # Common Utilities
│   │   ├── __init__.py
│   │   ├── config.py        # Storage configuration
│   │   ├── exceptions.py    # Custom exceptions
│   │   └── utils.py         # Shared utilities
│   │
│   └── manager.py          # Main storage manager
│
├── config/                 # Configuration Files
│   ├── redis_config.yaml    # Redis configuration
│   ├── postgres_config.yaml # PostgreSQL configuration
│   └── store_config.yaml    # Model store configuration
│ 
├── tests/                 # Test Suite
│   ├── redis/
│   ├── postgresql/ 
│   ├── model_store/
│   └── test_manager.py
│
├── scripts/              # Utility Scripts
│   ├── setup_db.py       # Database setup
│   ├── migrate.py        # Run migrations
│   └── cleanup.py        # Cleanup utilities
│
├── requirements.txt     # Dependencies
└── setup.py            # Package setup






gpu-faas-project/
├── gpu-infrastructure/        # Physical GPU layer
│   └── [Previous GPU infrastructure structure]
│
├── gpu-management/           # GPU Management layer
│   └── [Previous GPU management structure]  
│
├── gpu-faas/                # FaaS Application layer
│   └── [Previous GPU FaaS structure]
│
├── storage/                 # Storage layer
│   └── [Previous Storage layer structure]
│
├── integration/            # Integration Layer
│   ├── src/
│   │   ├── coordinator.py  # Main system coordinator
│   │   ├── interfaces/     # Layer interfaces
│   │   └── config.py       # Integration config
│   └── tests/
│
├── config/                # Project-wide configs
│   ├── system.yaml
│   └── integration.yaml
│
├── scripts/              # Project scripts
│   ├── setup.sh
│   └── deploy.sh
│
├── docker/              # Docker files
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── docs/               # Documentation
├── requirements.txt    # Dependencies
└── README.md          # Project info