#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化工具 - 监控和优化系统性能
"""

import time
import psutil
import torch
from typing import Dict, Any
from functools import wraps

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
        
    def start_timer(self, name: str):
        """开始计时"""
        self.metrics[name] = {
            'start_time': time.time(),
            'end_time': None,
            'duration': None
        }
        
    def end_timer(self, name: str):
        """结束计时"""
        if name in self.metrics:
            self.metrics[name]['end_time'] = time.time()
            self.metrics[name]['duration'] = (
                self.metrics[name]['end_time'] - self.metrics[name]['start_time']
            )
            
    def get_duration(self, name: str) -> float:
        """获取持续时间"""
        return self.metrics.get(name, {}).get('duration', 0)
        
    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存
            'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存
            'percent': process.memory_percent()
        }
        
    def get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        return psutil.cpu_percent(interval=0.1)
        
    def get_gpu_usage(self) -> Dict[str, Any]:
        """获取GPU使用情况"""
        if not torch.cuda.is_available():
            return {'available': False}
            
        return {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'current_memory_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'max_memory_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
            'memory_cached_mb': torch.cuda.memory_cached() / 1024 / 1024
        }
        
    def print_stats(self):
        """打印性能统计"""
        print("\n" + "="*50)
        print("性能统计")
        print("="*50)
        
        # 时间统计
        for name, metric in self.metrics.items():
            if metric['duration'] is not None:
                print(f"{name}: {metric['duration']:.3f}秒")
                
        # 内存统计
        memory = self.get_memory_usage()
        print(f"内存使用: {memory['rss_mb']:.1f}MB (物理), {memory['vms_mb']:.1f}MB (虚拟)")
        
        # CPU统计
        cpu = self.get_cpu_usage()
        print(f"CPU使用率: {cpu}%")
        
        # GPU统计
        gpu = self.get_gpu_usage()
        if gpu['available']:
            print(f"GPU内存: {gpu['current_memory_allocated_mb']:.1f}MB (当前)")
            print(f"GPU最大内存: {gpu['max_memory_allocated_mb']:.1f}MB (峰值)")
        else:
            print("GPU: 不可用")
            
        print("="*50)

def time_it(func):
    """计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = PerformanceMonitor()
        monitor.start_timer(func.__name__)
        result = func(*args, **kwargs)
        monitor.end_timer(func.__name__)
        print(f"{func.__name__} 执行时间: {monitor.get_duration(func.__name__):.3f}秒")
        return result
    return wrapper

class BatchProcessor:
    """批处理优化器"""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.monitor = PerformanceMonitor()
        
    def process_in_batches(self, data: list, process_func):
        """分批处理数据"""
        results = []
        total_batches = (len(data) + self.batch_size - 1) // self.batch_size
        
        self.monitor.start_timer('total_processing')
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_results = process_func(batch)
            results.extend(batch_results)
            
            # 打印进度
            if (i // self.batch_size) % 10 == 0:
                print(f"处理进度: {min(i + self.batch_size, len(data))}/{len(data)}")
                
        self.monitor.end_timer('total_processing')
        return results
        
    def optimize_batch_size(self, data: list, process_func, min_size=8, max_size=64, step=8):
        """自动优化批处理大小"""
        best_size = self.batch_size
        best_time = float('inf')
        
        print("正在优化批处理大小...")
        
        for size in range(min_size, max_size + 1, step):
            self.batch_size = size
            self.monitor.start_timer(f'batch_size_{size}')
            
            # 处理小样本测试性能
            test_data = data[:min(100, len(data))]
            self.process_in_batches(test_data, process_func)
            
            self.monitor.end_timer(f'batch_size_{size}')
            duration = self.monitor.get_duration(f'batch_size_{size}')
            
            print(f"批处理大小 {size}: {duration:.3f}秒")
            
            if duration < best_time:
                best_time = duration
                best_size = size
                
        print(f"最优批处理大小: {best_size}")
        self.batch_size = best_size
        return best_size

# 使用示例
if __name__ == "__main__":
    # 创建性能监控器
    monitor = PerformanceMonitor()
    
    # 监控示例操作
    monitor.start_timer('example_operation')
    time.sleep(1)  # 模拟耗时操作
    monitor.end_timer('example_operation')
    
    # 打印性能统计
    monitor.print_stats()
    
    # 测试批处理
    def dummy_process(batch):
        """模拟处理函数"""
        time.sleep(0.01)  # 模拟处理时间
        return [f"processed_{item}" for item in batch]
        
    processor = BatchProcessor(batch_size=16)
    test_data = list(range(100))
    
    # 优化批处理大小
    optimal_size = processor.optimize_batch_size(test_data, dummy_process)
    
    # 使用最优批处理大小处理数据
    results = processor.process_in_batches(test_data, dummy_process)
    print(f"处理完成，共 {len(results)} 条结果")