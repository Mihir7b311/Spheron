# memory_tracker/utils.py

def bytes_to_mb(bytes_val: int) -> float:
    """Convert bytes to megabytes"""
    return bytes_val / (1024 * 1024)

def mb_to_bytes(mb_val: float) -> int:
    """Convert megabytes to bytes"""
    return int(mb_val * 1024 * 1024)

def calculate_memory_percentage(used: int, total: int) -> float:
    """Calculate memory usage percentage"""
    return (used / total) * 100