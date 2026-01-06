import torch

class MemoryManager:
    def __init__(self, config, device="cuda"):
        _, total_gpu_memory = torch.cuda.mem_get_info()
        self.pool_size = int(total_gpu_memory * config.gpu_memory_utilization)
        self.device = device
        # 1. 申请全局大内存池 (不初始化内容)
        self.memory_pool = torch.empty(self.pool_size, dtype=torch.uint8, device=device)
        self.offset = 0 # 当前分配指针
        
        self.kv_buffer = None
        self.total_kv_blocks = 0

    def allocate_weights(self, num_bytes: int):
        """从池子中切出一块用于存放模型权重"""
        # 256字节对齐
        while self.offset % 256 != 0:
            self.offset += 1
            
        if self.offset + num_bytes > self.pool_size:
            used_mb = self.offset / 1024**2
            req_mb = num_bytes / 1024**2
            total_mb = self.pool_size / 1024**2
            raise RuntimeError(f"OOM: 显存池不足以分配权重。已用: {used_mb:.2f}MB, 请求: {req_mb:.2f}MB, 总计: {total_mb:.2f}MB")
            
        # 切片并移动指针
        ptr = self.memory_pool[self.offset : self.offset + num_bytes]
        self.offset += num_bytes
        return ptr

    def init_kv_pool(self, block_bytes_per_block: int):
        """权重加载完毕后，将剩余显存划分为 KV Cache 池"""
        remaining_bytes = self.pool_size - self.offset
        
        # 计算能放下多少个 block
        self.total_kv_blocks = remaining_bytes // block_bytes_per_block
        
        if self.total_kv_blocks <= 0:
            raise RuntimeError(f"没有足够的显存用于 KV Cache。剩余: {remaining_bytes/1024**2:.2f} MB")
            
        # 锁定 KV 区域
        kv_end = self.offset + self.total_kv_blocks * block_bytes_per_block
        self.kv_buffer = self.memory_pool[self.offset : kv_end]
        
        print(f"\n{'='*40}")
        print(f"Global Memory Pool Report")
        print(f"{'='*40}")
        print(f"  Total Capacity : {self.pool_size / 1024**3:.2f} GB")
        print(f"  Weights Used   : {self.offset / 1024**3:.2f} GB")
        print(f"  KV Cache Area  : {len(self.kv_buffer) / 1024**3:.2f} GB")
        print(f"  Shared Blocks  : {self.total_kv_blocks}")
        print(f"{'='*40}\n")
        
        return self.total_kv_blocks

    def get_kv_buffer(self):
        if self.kv_buffer is None:
            raise RuntimeError("KV Pool not initialized. Call init_kv_pool() first.")
        return self.kv_buffer
