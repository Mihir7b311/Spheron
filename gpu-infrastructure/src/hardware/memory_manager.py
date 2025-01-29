from typing import Dict, Optional, List
import logging
from dataclasses import dataclass
import time

@dataclass
class MemorySegment:
    """Represents GPU memory segment"""
    segment_id: str
    start_address: int
    size: int
    in_use: bool = False
    owner_id: Optional[str] = None
    allocation_time: float = 0.0

class MemoryManager:
    """Manages GPU memory allocation"""
    
    def __init__(self, gpu_id: int, total_memory: int):
        self.gpu_id = gpu_id
        self.total_memory = total_memory
        self.segments: Dict[str, MemorySegment] = {}
        self.free_memory = total_memory
        self.next_segment_id = 0
        self.min_segment_size = 256 * 1024 * 1024  # 256MB minimum

    def allocate_memory(self, size: int, owner_id: str) -> Optional[MemorySegment]:
        """Allocate memory segment"""
        try:
            if size < self.min_segment_size:
                size = self.min_segment_size
                
            if size > self.free_memory:
                return None
                
            # Find suitable location
            start_address = self._find_free_space(size)
            if start_address is None:
                return None
                
            # Create segment
            segment_id = f"seg_{self.next_segment_id}"
            self.next_segment_id += 1
            
            segment = MemorySegment(
                segment_id=segment_id,
                start_address=start_address,
                size=size,
                in_use=True,
                owner_id=owner_id,
                allocation_time=time.time()
            )
            
            self.segments[segment_id] = segment
            self.free_memory -= size
            
            logging.info(
                f"Allocated {size} bytes of memory on GPU {self.gpu_id} "
                f"for {owner_id} (segment {segment_id})"
            )
            
            return segment
            
        except Exception as e:
            logging.error(f"Memory allocation failed: {e}")
            return None

    def free_memory_segment(self, segment_id: str) -> bool:
        """Free memory segment"""
        try:
            if segment_id not in self.segments:
                return False
                
            segment = self.segments[segment_id]
            self.free_memory += segment.size
            del self.segments[segment_id]
            
            logging.info(
                f"Freed memory segment {segment_id} "
                f"({segment.size} bytes) on GPU {self.gpu_id}"
            )
            return True
            
        except Exception as e:
            logging.error(f"Failed to free memory segment: {e}")
            return False

    def _find_free_space(self, size: int) -> Optional[int]:
        """Find free memory space"""
        if not self.segments:
            return 0
            
        # Sort segments by address
        sorted_segments = sorted(
            self.segments.values(),
            key=lambda x: x.start_address
        )
        
        # Check gaps between segments
        prev_end = 0
        for segment in sorted_segments:
            if segment.start_address - prev_end >= size:
                return prev_end
            prev_end = segment.start_address + segment.size
            
        # Check space after last segment
        if self.total_memory - prev_end >= size:
            return prev_end
            
        return None

    async def defragment(self) -> None:
        """Defragment memory"""
        try:
            if not self._needs_defragmentation():
                return
                
            logging.info(f"Starting memory defragmentation on GPU {self.gpu_id}")
            
            # Sort segments by address
            sorted_segments = sorted(
                self.segments.values(),
                key=lambda x: x.start_address
            )
            
            # Compact segments
            current_address = 0
            for segment in sorted_segments:
                if segment.start_address != current_address:
                    # Move segment
                    segment.start_address = current_address
                current_address += segment.size
                
            logging.info("Memory defragmentation completed")
            
        except Exception as e:
            logging.error(f"Defragmentation failed: {e}")

    def _needs_defragmentation(self) -> bool:
        """Check if defragmentation needed"""
        if not self.segments:
            return False
            
        # Calculate fragmentation ratio
        used_memory = sum(s.size for s in self.segments.values())
        sorted_segments = sorted(
            self.segments.values(),
            key=lambda x: x.start_address
        )
        last_address = sorted_segments[-1].start_address + sorted_segments[-1].size
        
        fragmentation = 1 - (used_memory / last_address)
        return fragmentation > 0.2  # 20% fragmentation threshold

    def get_memory_map(self) -> List[Dict]:
        """Get current memory map"""
        return [{
            "segment_id": s.segment_id,
            "start_address": s.start_address,
            "size": s.size,
            "in_use": s.in_use,
            "owner_id": s.owner_id
        } for s in sorted(
            self.segments.values(),
            key=lambda x: x.start_address
        )]

    def get_stats(self) -> Dict:
        """Get memory statistics"""
        return {
            "total_memory": self.total_memory,
            "free_memory": self.free_memory,
            "used_memory": self.total_memory - self.free_memory,
            "segment_count": len(self.segments),
            "utilization": (self.total_memory - self.free_memory) / self.total_memory
        }