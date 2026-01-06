import torch

class MemoryManager:
    def __init__(self, config, device="cuda"):
        _, total_gpu_memory = torch.cuda.mem_get_info()
        self.pool_size = int(total_gpu_memory * config.gpu_memory_utilization)
        self.device = device
        
        # 1. 申请全局大内存池 (一次性占位)
        print(f"[MemoryManager] Allocating global pool: {self.pool_size / 1024**3:.2f} GB")
        self.memory_pool = torch.empty(self.pool_size, dtype=torch.uint8, device=device)
        
        # --- 内存管理数据结构 (空闲链表) ---
        # 初始时，整个池子都是空闲的: [(start, end)]
        self.free_segments = [(0, self.pool_size)] 
        
        # 记录每个模型占用的内存片段: { model_id: [(start, size), ...] }
        self.model_allocations = {}
        
        # 当前正在加载的模型ID (上下文状态)
        self.current_loading_model_id = None
        
        self.kv_buffer = None
        self.total_kv_blocks = 0

    def start_allocation(self, model_id):
        """开始为某个模型分配内存，后续的 allocate_weights 都会记在这个 ID 下"""
        self.current_loading_model_id = model_id
        if model_id not in self.model_allocations:
            self.model_allocations[model_id] = []
        print(f"[MemoryManager] >>> Start tracking allocations for: {model_id}")

    def stop_allocation(self):
        """停止记录"""
        print(f"[MemoryManager] <<< Stop tracking allocations for: {self.current_loading_model_id}")
        self.current_loading_model_id = None

    def allocate_weights(self, num_bytes: int):
        """从空闲段中切出一块 (First-Fit 算法)"""
        # 256字节对齐
        remainder = num_bytes % 256
        if remainder != 0:
            num_bytes += (256 - remainder)
            
        best_segment_idx = -1
        alloc_start = -1

        # 1. First-Fit: 寻找第一个能塞下的空闲段
        for i, (start, end) in enumerate(self.free_segments):
            if (end - start) >= num_bytes:
                best_segment_idx = i
                alloc_start = start
                break
        
        if best_segment_idx == -1:
            self._print_fragmentation_info()
            raise RuntimeError(f"OOM: 无法找到连续 {num_bytes/1024**2:.2f}MB 的空间用于分配权重。")

        # 2. 更新空闲链表
        segment_start, segment_end = self.free_segments[best_segment_idx]
        
        # 如果这块空间刚好用完，直接移除该段
        if (segment_end - segment_start) == num_bytes:
            self.free_segments.pop(best_segment_idx)
        else:
            # 否则，修改起始位置，剩下的留作他用
            self.free_segments[best_segment_idx] = (segment_start + num_bytes, segment_end)

        # 3. 记录归属（如果正在追踪）
        if self.current_loading_model_id is not None:
            # 优化：如果这一次分配紧挨着上一次分配，合并记录（减少列表长度）
            allocs = self.model_allocations[self.current_loading_model_id]
            if allocs and (allocs[-1][0] + allocs[-1][1] == alloc_start):
                allocs[-1] = (allocs[-1][0], allocs[-1][1] + num_bytes)
            else:
                allocs.append((alloc_start, num_bytes))

        # 4. 返回 Tensor 切片
        return self.memory_pool[alloc_start : alloc_start + num_bytes]

    def free_model(self, model_id):
        """释放指定模型的权重，产生新的空闲段（洞）"""
        if model_id not in self.model_allocations:
            print(f"Warning: Model {model_id} not found or already freed.")
            return
            
        print(f"\n[MemoryManager] !!! Freeing model: {model_id} !!!")
        chunks = self.model_allocations[model_id]
        
        # 将归还的内存块加回 free_segments
        for (start, size) in chunks:
            self.free_segments.append((start, start + size))
            
        del self.model_allocations[model_id]
        
        # 碎片整理：合并相邻的空闲段
        self._merge_free_segments()
        self._print_fragmentation_info()

    def _merge_free_segments(self):
        """合并相邻的空闲区间"""
        self.free_segments.sort(key=lambda x: x[0])
        merged = []
        if not self.free_segments:
            return
            
        curr_start, curr_end = self.free_segments[0]
        for i in range(1, len(self.free_segments)):
            next_start, next_end = self.free_segments[i]
            if next_start == curr_end: # 相邻，合并
                curr_end = next_end
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged.append((curr_start, curr_end))
        self.free_segments = merged

    def init_kv_pool(self, block_bytes_per_block: int):
        """
        初始化 KV Cache 池。
        KV Cache 需要物理连续的大块内存，因此我们取当前【最大】的一块空闲段。
        """
        if not self.free_segments:
             raise RuntimeError("No memory left for KV Cache.")
             
        # 找最大的空闲段
        max_idx, max_len = -1, 0
        for i, (start, end) in enumerate(self.free_segments):
            length = end - start
            if length > max_len:
                max_len = length
                max_idx = i
                
        if max_idx == -1:
             raise RuntimeError("No valid free segment found.")

        # 锁定这块区域
        start, end = self.free_segments[max_idx]
        
        # 计算能放下的 Block 数量
        self.total_kv_blocks = max_len // block_bytes_per_block
        if self.total_kv_blocks <= 0:
             raise RuntimeError(f"KV Cache space too small: {max_len/1024**2:.2f} MB")

        kv_bytes = self.total_kv_blocks * block_bytes_per_block
        used_end = start + kv_bytes
        
        # 切片出 KV Buffer
        self.kv_buffer = self.memory_pool[start : used_end]
        
        # 更新空闲段 (吃掉这块空间)
        if used_end < end:
            self.free_segments[max_idx] = (used_end, end)
        else:
            self.free_segments.pop(max_idx)

        print(f"\n{'='*40}")
        print(f"KV Pool Initialized")
        print(f"{'='*40}")
        print(f"  Source Segment : {start}-{end} (Size: {max_len/1024**3:.2f} GB)")
        print(f"  KV Capacity    : {self.total_kv_blocks} blocks")
        print(f"  Remaining Free : {len(self.free_segments)} segments")
        print(f"{'='*40}\n")
        
        return self.total_kv_blocks

    def get_kv_buffer(self):
        if self.kv_buffer is None:
            raise RuntimeError("KV Pool not initialized. Call init_kv_pool() first.")
        return self.kv_buffer
        
    def _print_fragmentation_info(self):
        free_total = sum(end-start for start, end in self.free_segments)
        print(f"  > [Memory Status] Free: {free_total/1024**2:.1f}MB | Segments: {len(self.free_segments)} chunks")