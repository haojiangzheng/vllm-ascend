import torch
try:
    import torch_npu
except ImportError:
    torch_npu = None
import time
import math
import numpy as np
import os
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import json

class ProfilerBackend(Enum):
    """Profiler后端类型"""
    NPU = "npu"
    CUDA = "cuda"

@dataclass
class ProfilerConfig:
    """Profiler配置类"""
    backend: ProfilerBackend
    trace_file_path: Optional[str] = None
    kernel_names: Optional[List[str]] = None
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    experimental_config: Optional[Dict[str, Any]] = None
    # Schedule 配置
    schedule_wait: int = 0
    schedule_warmup: int = 0
    schedule_active: int = 20
    schedule_repeat: int = 1
    schedule_skip_first: int = 1

class ProfilerFactory:
    """Profiler工厂类 - 根据后端类型创建相应的profiler"""
    
    @staticmethod
    def create_profiler(config: ProfilerConfig):
        """创建profiler实例 - 简化版本，参考run_unified_profile_test的实现
        
        Args:
            config: Profiler配置
            
        Returns:
            profiler实例
        """
        if config.backend == ProfilerBackend.NPU:
            return ProfilerFactory._create_npu_profiler(config)
        elif config.backend == ProfilerBackend.CUDA:
            return ProfilerFactory._create_cuda_profiler(config)
        else:
            raise ValueError(f"不支持的profiler后端: {config.backend}")
    
    @staticmethod
    def _create_npu_profiler(config: ProfilerConfig):
        """创建NPU profiler - 完整版本"""
        try:
            import torch_npu
            
            # 处理实验性配置
            exp_config = config.experimental_config or {}
            profile_level = exp_config.get('profile_level', 'Level1')
            aic_metrics = exp_config.get('aic_metrics', 'PipeUtilization')
            export_type = exp_config.get('export_type', 'Text')
            
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                profiler_level=getattr(torch_npu.profiler.ProfilerLevel, profile_level, torch_npu.profiler.ProfilerLevel.Level1),
                aic_metrics=getattr(torch_npu.profiler.AiCMetrics, aic_metrics, torch_npu.profiler.AiCMetrics.PipeUtilization),
                export_type=getattr(torch_npu.profiler.ExportType, export_type, torch_npu.profiler.ExportType.Text)
            )
            
            profiler_kwargs = {
                'activities': [
                    torch_npu.profiler.ProfilerActivity.NPU,
                    torch_npu.profiler.ProfilerActivity.CPU
                ],
                'with_stack': config.with_stack,
                'record_shapes': config.record_shapes,
                'profile_memory': config.profile_memory,
                'experimental_config': experimental_config,
                'schedule': torch_npu.profiler.schedule(
                    wait=config.schedule_wait,
                    warmup=config.schedule_warmup,
                    active=config.schedule_active,
                    repeat=config.schedule_repeat,
                    skip_first=config.schedule_skip_first
                )
            }
            
            # 如果有trace文件路径，添加处理器
            if config.trace_file_path:
                profiler_kwargs['on_trace_ready'] = torch_npu.profiler.tensorboard_trace_handler(
                    config.trace_file_path, 
                )
            
            return torch_npu.profiler.profile(**profiler_kwargs)
            
        except ImportError:
            raise ImportError("torch_npu未安装，无法使用NPU profiler")
    
    @staticmethod
    def _create_cuda_profiler(config: ProfilerConfig):
        """创建CUDA profiler - 完整版本"""
        try:
            profiler_kwargs = {
                'activities': [
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
                'record_shapes': config.record_shapes,
                'profile_memory': config.profile_memory,
                'with_stack': config.with_stack,
                'schedule': torch.profiler.schedule(
                    wait=config.schedule_wait,
                    warmup=config.schedule_warmup,
                    active=config.schedule_active,
                    repeat=config.schedule_repeat
                )
            }
            
            # 如果有trace文件路径，添加处理器
            if config.trace_file_path:
                profiler_kwargs['on_trace_ready'] = torch.profiler.tensorboard_trace_handler(
                    config.trace_file_path
                )
            
            return torch.profiler.profile(**profiler_kwargs)
            
        except Exception as e:
            raise RuntimeError(f"创建CUDA profiler失败: {e}")
    

class PrecisionType(Enum):
    """支持的精度类型"""
    FP16 = torch.float16
    BF16 = torch.bfloat16
    FP32 = torch.float32
    INT8 = torch.int8

class DeviceType(Enum):
    """支持的设备类型"""
    CPU = "cpu"
    NPU = "npu"
    GPU = "cuda"

@dataclass
class AccuracyMetrics:
    """精度测试指标"""
    max_abs_error: float
    mean_abs_error: float
    max_rel_error: float
    mean_rel_error: float
    cosine_similarity: float
    mse: float
    rmse: float
    precision_type: str
    operator_name: str
    
    def __str__(self):
        return (f"精度指标 ({self.operator_name} - {self.precision_type}):\n"
                f"  最大绝对误差: {self.max_abs_error:.6e}\n"
                f"  平均绝对误差: {self.mean_abs_error:.6e}\n"
                f"  最大相对误差: {self.max_rel_error:.6e}\n"
                f"  平均相对误差: {self.mean_rel_error:.6e}\n"
                f"  余弦相似度: {self.cosine_similarity:.6f}\n"
                f"  MSE: {self.mse:.6e}\n"
                f"  RMSE: {self.rmse:.6e}")


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    avg_time_ms: float
    throughput: Optional[float]  # 可选的吞吐量指标
    precision_type: str
    device_type: str
    operator_name: str
    iterations: int = 20  # 测试迭代次数
    throughput_ops_per_sec: Optional[float] = None  # 每秒操作数吞吐量
    tops: Optional[float] = None  # TOPS (每秒万亿次操作)
    bandwidth_gb_s: Optional[float] = None  # 内存带宽 (GB/s)
    
    def __post_init__(self):
        """初始化后处理，确保throughput_ops_per_sec有值"""
        if self.throughput_ops_per_sec is None and self.throughput is not None:
            self.throughput_ops_per_sec = self.throughput
    
    def __str__(self):
        result = (f"性能指标 ({self.operator_name} - {self.device_type} - {self.precision_type}):\n"
                 f"  平均时间: {self.avg_time_ms:.3f}ms (批量执行模式)\n")
        
        if self.tops is not None:
            result += f"  计算性能: {self.tops:.2f} TOPS\n"
        
        if self.bandwidth_gb_s is not None:
            result += f"  内存带宽: {self.bandwidth_gb_s:.2f} GB/s\n"
        
        if self.throughput is not None:
            result += f"  吞吐量: {self.throughput:.2f} ops/s\n"
        
        return result


class BaseOperatorTest(ABC):
    """算子测试基类"""
    
    def __init__(self, operator_name: str):
        self.operator_name = operator_name
        self.supported_precisions = [PrecisionType.FP16, PrecisionType.BF16, PrecisionType.FP32]
        self.supported_devices = [DeviceType.CPU, DeviceType.NPU]
    
    @abstractmethod
    def generate_test_data(self, **kwargs) -> Dict[str, Any]:
        """生成测试数据"""
        pass
    
    @abstractmethod
    def run_cpu_reference(self, data: Dict[str, Any]) -> torch.Tensor:
        """运行CPU参考实现"""
        pass
    
    @abstractmethod
    def run_device_implementation(
        self, 
        data: Dict[str, Any], 
        device: str, 
        precision: PrecisionType,
        implementation: str = "default"
    ) -> torch.Tensor:
        """运行设备实现"""
        pass
    
    @abstractmethod
    def get_available_implementations(self, device: str) -> List[str]:
        """获取可用的实现方式"""
        pass
    
    def run_core_operator(
        self, 
        data: Dict[str, Any], 
        device: str, 
        precision: PrecisionType,
        implementation: str = "default"
    ) -> torch.Tensor:
        """运行核心算子操作（用于精确性能测试）
        
        默认实现调用run_device_implementation，子类可以重写此方法
        来只执行核心算子操作，排除数据预处理和后处理的时间开销
        """
        return self.run_device_implementation(data, device, precision, implementation)
    

    
    def calculate_throughput(self, data: Dict[str, Any], time_ms: float) -> Optional[float]:
        """计算吞吐量
        
        Args:
            data: 测试数据
            time_ms: 执行时间（毫秒）
            
        Returns:
            Optional[float]: 吞吐量，如果无法计算则返回None
        """
        # 默认实现返回None，子类应该重写此方法
        return None
    

class OperatorTestFramework:
    """算子测试框架"""
    
    def __init__(self, result_dir: str = "test_results"):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.operators: Dict[str, BaseOperatorTest] = {}
    
    def _measure_execution_time(self, func, device: str, num_iterations: int = 1) -> float:
        """使用设备事件精确测量执行时间
        
        Args:
            func: 要测量的函数
            device: 设备类型
            num_iterations: 测量次数
            
        Returns:
            List[float]: 每次执行的时间（毫秒）
        """
        times = []
        
        try:
            if "npu" in device:
                # NPU事件计时 - 批量执行版本
                start_event = torch_npu.npu.Event(enable_timing=True)
                end_event = torch_npu.npu.Event(enable_timing=True)
                
                # 初始同步确保设备就绪
                torch_npu.npu.synchronize()
                
                # 记录开始时间
                start_event.record()
                
                # 执行所有迭代
                for i in range(num_iterations):
                    func()
                
                # 确保所有NPU操作完成
                torch_npu.npu.synchronize()
                
                # 记录结束时间
                end_event.record()
                
                # 等待结束事件完成并计算总时间
                end_event.synchronize()
                total_time = start_event.elapsed_time(end_event)  # 毫秒
                
                # 计算平均时间
                avg_time = total_time / num_iterations
                return avg_time
                    
            elif "cuda" in device:
                # CUDA事件计时 - 批量执行版本
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # 初始同步确保设备就绪
                torch.cuda.synchronize()
                
                # 记录开始时间
                start_event.record()
                
                # 执行所有迭代
                for i in range(num_iterations):
                    func()
                
                # 确保所有CUDA操作完成
                torch.cuda.synchronize()
                
                # 记录结束时间
                end_event.record()
                
                # 等待结束事件完成并计算总时间
                end_event.synchronize()
                total_time = start_event.elapsed_time(end_event)  # 毫秒
                
                # 计算平均时间
                avg_time = total_time / num_iterations
                return avg_time
                    
            else:
                # CPU计时 - 批量执行版本
                import time as time_module
                
                # 记录开始时间
                start_time = time_module.perf_counter()
                
                # 执行所有迭代
                for i in range(num_iterations):
                    func()
                
                # 记录结束时间
                end_time = time_module.perf_counter()
                
                # 计算总时间和平均时间
                total_time = (end_time - start_time) * 1000  # 转换为毫秒
                avg_time = total_time / num_iterations
                return avg_time
                        
        except Exception as e:
            print(f"    ❌ 计时过程中发生错误: {str(e)}")
            raise
    
    def _measure_execution_time_v2(self, test_functions: List[Callable], device: str) -> float:
        """测量执行时间 V2版本（使用预先准备的函数列表）
        
        Args:
            test_functions: 预先准备的测试函数列表，每个函数对应一次迭代
            device: 设备类型
            
        Returns:
            float: 平均执行时间（毫秒）
        """
        num_iterations = len(test_functions)
        
        try:
            if "npu" in device:
                # NPU事件计时 - 使用预准备函数列表
                start_event = torch_npu.npu.Event(enable_timing=True)
                end_event = torch_npu.npu.Event(enable_timing=True)
                
                # 初始同步确保设备就绪
                torch_npu.npu.synchronize()
                
                # 记录开始时间
                start_event.record()
                
                # 执行所有预准备的函数
                for func in test_functions:
                    func()
                
                # 确保所有NPU操作完成
                torch_npu.npu.synchronize()
                
                # 记录结束时间
                end_event.record()
                
                # 等待结束事件完成并计算总时间
                end_event.synchronize()
                total_time = start_event.elapsed_time(end_event)  # 毫秒
                
                # 计算平均时间
                avg_time = total_time / num_iterations
                return avg_time
                    
            elif "cuda" in device:
                # CUDA事件计时 - 使用预准备函数列表
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # 初始同步确保设备就绪
                torch.cuda.synchronize()
                
                # 记录开始时间
                start_event.record()
                
                # 执行所有预准备的函数
                for func in test_functions:
                    func()
                
                # 确保所有CUDA操作完成
                torch.cuda.synchronize()
                
                # 记录结束时间
                end_event.record()
                
                # 等待结束事件完成并计算总时间
                end_event.synchronize()
                total_time = start_event.elapsed_time(end_event)  # 毫秒
                
                # 计算平均时间
                avg_time = total_time / num_iterations
                return avg_time
                    
            else:
                # CPU计时 - 使用预准备函数列表
                import time as time_module
                
                # 记录开始时间
                start_time = time_module.perf_counter()
                
                # 执行所有预准备的函数
                for func in test_functions:
                    func()
                
                # 记录结束时间
                end_time = time_module.perf_counter()
                
                # 计算总时间和平均时间
                total_time = (end_time - start_time) * 1000  # 转换为毫秒
                avg_time = total_time / num_iterations
                return avg_time
                        
        except Exception as e:
            print(f"    ❌ V2计时过程中发生错误: {str(e)}")
            raise
    
    def run_unified_profile_test(
        self,
        operator_name: str,
        test_data: Dict[str, Any],
        device: str = "npu:0",
        precision: PrecisionType = PrecisionType.BF16,
        num_warmup: int = 5,
        num_iterations: int = 10,
        enable_profile: bool = True,
        test_name: str = "unified_profile",
        profile_level: str = "Level1",
        aic_metrics: str = "PipeUtilization",
        export_type: str = "Text",
        **kwargs
    ) -> Dict[str, Any]:
        """运行统一profile测试 
        
        Args:
            operator_name: 算子名称
            test_data: 测试数据
            device: 设备
            precision: 精度类型
            num_warmup: 预热次数
            num_iterations: 迭代次数
            enable_profile: 是否启用profile
            test_name: 测试名称
            profile_level: Profile级别
            aic_metrics: AIC指标类型
            export_type: 导出类型
            **kwargs: 其他参数
        """
        
        if operator_name not in self.operators:
            raise ValueError(f"算子 {operator_name} 未注册")
        
        operator_test = self.operators[operator_name]
        
        print(f"\n{'='*60}")
        print(f"统一Profile测试: {operator_name} - {test_name}")
        print(f"设备: {device}, 精度: {precision.name}, 迭代次数: {num_iterations}")
        print(f"所有实现的profile将合并到一个文件中")
        print(f"{'='*60}")
        
        implementations = operator_test.get_available_implementations(device)
        comparison_results = {}
        
        # 为每个实现分别进行profile测试
        for impl_name in implementations:
            print(f"\n--- 测试实现: {impl_name} ---")
            
            try:
                # 为当前实现创建独立的profile保存路径
                prof_save_path = None
                prof = None
                
                if enable_profile:
                    prof_save_path = self.result_dir / f"{operator_name}_{impl_name}_{test_name}_{precision.name}_profile"
                    prof_save_path.mkdir(exist_ok=True)
                    
                    # 使用 ProfilerFactory 创建 profiler
                    backend = ProfilerBackend.NPU if device.startswith("npu") else ProfilerBackend.CUDA
                    
                    if device.startswith("npu"):
                        # NPU 配置
                        config = ProfilerConfig(
                            backend=backend,
                            trace_file_path=str(prof_save_path),
                            record_shapes=False,
                            profile_memory=False,
                            with_stack=False,
                            schedule_wait=0,
                            schedule_warmup=0,
                            schedule_active=num_iterations,
                            schedule_repeat=1,
                            schedule_skip_first=1,
                            experimental_config={
                                'profile_level': profile_level,
                                'aic_metrics': aic_metrics,
                                'export_type': export_type
                            }
                        )
                    else:
                        # CUDA 配置
                        config = ProfilerConfig(
                            backend=backend,
                            trace_file_path=str(prof_save_path),
                            record_shapes=True,
                            profile_memory=True,
                            with_stack=True,
                            schedule_wait=1,
                            schedule_warmup=1,
                            schedule_active=num_iterations,
                            schedule_repeat=2
                        )
                    
                    prof = ProfilerFactory.create_profiler(config)
                    
                # 执行profile测试
                print(f"开始执行 {num_iterations} 次迭代...")
                for i in range(num_warmup):
                    operator_test.run_device_implementation(
                        test_data, device, precision, impl_name
                    )

                prof.start()
                for i in range(num_iterations):
                    operator_test.run_device_implementation(
                        test_data, device, precision, impl_name
                    )
                    prof.step()
                
                # 设备同步
                if device.startswith("npu"):
                    torch_npu.npu.synchronize()
                elif device.startswith("cuda"):
                    torch.cuda.synchronize()
                
                # 停止当前实现的profile
                if prof is not None:
                    prof.stop()
                    print(f"✅ Profile已保存到: {prof_save_path}")
                
                # 记录结果
                comparison_results[impl_name] = {
                    "profile_path": prof_save_path if enable_profile else None,
                    "iterations": num_iterations,
                    "status": "success"
                }
                
                print(f"✅ 完成 {num_iterations} 次迭代")
                
            except Exception as e:
                print(f"❌ 实现 {impl_name} 测试失败: {str(e)}")
                comparison_results[impl_name] = {
                    "profile_path": None,
                    "iterations": 0,
                    "status": "failed",
                    "error": str(e)
                }
        
        # 保存对比结果
        results = {
            "operator_name": operator_name,
            "test_name": test_name,
            "device": device,
            "precision": precision.name,
            "num_iterations": num_iterations,
            "num_warmup": num_warmup,
            "profile_config": {
                "enable_profile": enable_profile,
                "profile_level": profile_level,
                "aic_metrics": aic_metrics,
                "export_type": export_type
            },
            "comparison_results": comparison_results,
            "test_data_info": test_data.get('metadata', {}),
            "profile_enabled": enable_profile
        }
        
        self.save_results(results, f"{operator_name}_{test_name}_unified_profile")
        
        print(f"\n{'='*60}")
        print("分别Profile测试完成")
        if enable_profile:
            print("每个实现的Profile文件:")
            for impl_name, result in comparison_results.items():
                if result.get("profile_path"):
                    print(f"  {impl_name}: {result['profile_path']}")
        print(f"{'='*60}")
        
        return results

    def register_operator(self, operator_test: BaseOperatorTest):
        """注册算子测试"""
        self.operators[operator_test.operator_name] = operator_test
        print(f"✓ 已注册算子: {operator_test.operator_name}")
    
    def calculate_accuracy_metrics(
        self, 
        reference: torch.Tensor, 
        test_output: torch.Tensor,
        operator_name: str,
        precision_type: str
    ) -> AccuracyMetrics:
        """计算精度指标"""
        
        # 确保tensor在CPU上
        reference = reference.cpu().float()
        test_output = test_output.cpu().float()
        
        # 计算绝对误差
        abs_error = torch.abs(reference - test_output)
        max_abs_error = torch.max(abs_error).item()
        mean_abs_error = torch.mean(abs_error).item()
        
        # 计算相对误差
        rel_error = abs_error / (torch.abs(reference) + 1e-8)
        max_rel_error = torch.max(rel_error).item()
        mean_rel_error = torch.mean(rel_error).item()
        
        # 计算余弦相似度
        reference_flat = reference.flatten()
        test_output_flat = test_output.flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            reference_flat.unsqueeze(0), 
            test_output_flat.unsqueeze(0)
        ).item()
        
        # 计算MSE和RMSE
        mse = torch.mean((reference - test_output) ** 2).item()
        rmse = math.sqrt(mse)
        
        return AccuracyMetrics(
            max_abs_error=max_abs_error,
            mean_abs_error=mean_abs_error,
            max_rel_error=max_rel_error,
            mean_rel_error=mean_rel_error,
            cosine_similarity=cosine_sim,
            mse=mse,
            rmse=rmse,
            precision_type=precision_type,
            operator_name=operator_name
        )
    
    def run_performance_test(
        self,
        operator_test: BaseOperatorTest,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
        num_warmup: int = 5,
        num_iterations: int = 20
    ) -> PerformanceMetrics:
        """运行性能测试"""
        
        print(f"  性能测试: {device} - {precision.name} - {implementation}")
        
        # 预热
        for _ in range(num_warmup):
            _ = operator_test.run_device_implementation(data, device, precision, implementation)
        
        # 同步设备
        if "npu" in device:
            torch_npu.npu.synchronize()
        elif "cuda" in device:
            torch.cuda.synchronize()
        
        # 性能测试 - 使用设备事件计时
        def run_once():
            return operator_test.run_device_implementation(data, device, precision, implementation)
        
        avg_time = self._measure_execution_time(run_once, device, num_iterations)
        
        # 计算吞吐量
        throughput = operator_test.calculate_throughput(data, avg_time)
        
        # 计算TOPS和带宽
        tops = None
        bandwidth_gb_s = None
        
        # 尝试计算TOPS
        if hasattr(operator_test, 'calculate_flops'):
            try:
                flops = operator_test.calculate_flops(data)
                if flops is not None and avg_time > 0:
                    # avg_time是毫秒，转换为秒后计算TOPS
                    avg_time_s = avg_time / 1000.0
                    tops = flops / (avg_time_s * 1e12)  # 转换为TOPS
            except Exception as e:
                print(f"警告: 计算TOPS时出错: {e}")
                pass
        
        # 尝试计算带宽
        if hasattr(operator_test, 'calculate_bandwidth'):
            try:
                bandwidth_gb_s = operator_test.calculate_bandwidth(data, avg_time)
            except Exception as e:
                print(f"警告: 计算带宽时出错: {e}")
                pass
        
        # 性能指标将在汇总时显示，这里不重复输出
        
        return PerformanceMetrics(
            avg_time_ms=avg_time,
            throughput=throughput,
            precision_type=precision.name,
            device_type=device,
            operator_name=operator_test.operator_name,
            iterations=num_iterations,
            throughput_ops_per_sec=throughput,
            tops=tops,
            bandwidth_gb_s=bandwidth_gb_s
        )
    
    def run_core_operator_performance_test_v2(
        self,
        operator_test: BaseOperatorTest,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
        num_warmup: int = 10,
        num_iterations: int = 20
    ) -> PerformanceMetrics:
        """运行核心算子性能测试 V2版本（预先准备所有数据，避免数据准备开销）
        
        这个版本会预先准备 num_warmup + num_iterations 份数据，
        确保性能测试时不会有任何数据准备的开销，获得更准确的算子性能数据。
        
        Args:
            operator_test: 算子测试实例
            data: 原始测试数据
            device: 设备类型
            precision: 精度类型
            implementation: 实现方式
            num_warmup: 预热次数
            num_iterations: 测试迭代次数
            
        Returns:
            PerformanceMetrics: 性能指标
        """
        
        print(f"  核心算子性能测试 V2: {device} - {precision.name} - {implementation}")
        
        # 检查算子是否支持分离的核心算子测试
        if not hasattr(operator_test, '_prepare_data_for_core_operator') or \
           not hasattr(operator_test, '_execute_core_operator'):
            print(f"    ⚠️  算子不支持分离的核心算子测试，使用完整方法")
            return self.run_performance_test(
                operator_test, data, device, precision, implementation, num_warmup, num_iterations
            )
        
        total_runs = num_warmup + num_iterations
        
        # 预先准备所有数据（不计入性能测试时间）
        print(f"    📋 预先准备所有测试数据 ({total_runs} 份)...")
        prepared_data_list = []
        
        for i in range(total_runs):
            # 为每次运行准备独立的数据副本，避免缓存效应
            prepared_data = operator_test._prepare_data_for_core_operator(data, device, precision, implementation)
            prepared_data_list.append(prepared_data)
            
            # 每准备10份数据显示一次进度
            if (i + 1) % 10 == 0 or i == 0:
                print(f"      进度: {i + 1}/{total_runs}")
        
        print(f"    ✅ 数据准备完成，共 {total_runs} 份")
        
        # 使用预热数据进行算子预热
        print(f"    🔥 算子预热 ({num_warmup} 次)...")
        for i in range(num_warmup):
            _ = operator_test._execute_core_operator(prepared_data_list[i], implementation)
            if i == 0:  # 第一次预热后同步一次
                if "npu" in device:
                    torch_npu.npu.synchronize()
                elif "cuda" in device:
                    torch.cuda.synchronize()
        
        # 最终同步，确保预热完成
        if "npu" in device:
            torch_npu.npu.synchronize()
        elif "cuda" in device:
            torch.cuda.synchronize()
        
        print(f"    ⏱️  开始性能测试 ({num_iterations} 次)...")
        
        # 性能测试 - 使用预先准备的数据
        test_data_list = prepared_data_list[num_warmup:]  # 使用预热后的数据进行测试
        
        # 创建测试函数列表
        test_functions = []
        for i in range(num_iterations):
            test_data = test_data_list[i]
            test_functions.append(lambda data=test_data: operator_test._execute_core_operator(data, implementation))
        
        # 使用设备事件计时，测试纯算子执行时间
        avg_time = self._measure_execution_time_v2(test_functions, device)
        
        print(f"    ✅ 核心算子平均时间: {avg_time:.3f}ms (V2版本 - 无数据准备开销)")
        
        # 计算吞吐量
        throughput = operator_test.calculate_throughput(data, avg_time)
        
        # 计算TOPS和带宽
        tops = None
        bandwidth_gb_s = None
        
        # 尝试计算TOPS
        if hasattr(operator_test, 'calculate_flops'):
            try:
                flops = operator_test.calculate_flops(data)
                if flops is not None and avg_time > 0:
                    # avg_time是毫秒，转换为秒后计算TOPS
                    avg_time_s = avg_time / 1000.0
                    tops = flops / (avg_time_s * 1e12)  # 转换为TOPS
            except Exception as e:
                print(f"警告: 计算TOPS时出错: {e}")
                pass
        
        # 尝试计算带宽
        if hasattr(operator_test, 'calculate_bandwidth'):
            try:
                bandwidth_gb_s = operator_test.calculate_bandwidth(data, avg_time)
            except Exception as e:
                print(f"警告: 计算带宽时出错: {e}")
                pass
        
        # 清理NPU内存，防止内存泄漏
        try:
            if "npu" in device:
                torch_npu.npu.empty_cache()
        except Exception:
            pass  # 忽略清理错误
        
        return PerformanceMetrics(
            avg_time_ms=avg_time,
            throughput=throughput,
            precision_type=precision.name,
            device_type=device,
            operator_name=f"{operator_test.operator_name}_core_v2",
            iterations=num_iterations,
            throughput_ops_per_sec=throughput,
            tops=tops,
            bandwidth_gb_s=bandwidth_gb_s
        )
    
    
    def run_accuracy_test(
        self,
        operator_name: str,
        test_data: Dict[str, Any],
        test_name: str = "default"
    ) -> Dict[str, Any]:
        """运行精度测试"""
        
        if operator_name not in self.operators:
            raise ValueError(f"算子 {operator_name} 未注册")
        
        operator_test = self.operators[operator_name]
        
        print(f"\n{'='*60}")
        print(f"精度测试: {operator_name} - {test_name}")
        print(f"{'='*60}")
        
        # 运行CPU参考实现
        print("运行CPU参考实现...")
        cpu_output = operator_test.run_cpu_reference(test_data)
        
        results = {
            'operator_name': operator_name,
            'test_name': test_name,
            'test_data_info': test_data.get('metadata', {}),
            'cpu_output_shape': list(cpu_output.shape),
            'precision_results': {}
        }
        
        # 对每种精度和设备进行测试
        for precision in operator_test.supported_precisions:
            precision_name = precision.name
            print(f"\n=== 测试精度: {precision_name} ===")
            
            precision_results = {}
            
            for device_type in operator_test.supported_devices:
                if device_type == DeviceType.CPU:
                    continue  # CPU已经作为参考实现
                
                device = device_type.value
                if device == "npu":
                    device = "npu:0"
                elif device == "cuda":
                    device = "cuda:0"
                
                device_results = {}
                implementations = operator_test.get_available_implementations(device)
                
                for impl in implementations:
                    try:
                        print(f"  运行 {device} - {impl} 实现...")
                        device_output = operator_test.run_device_implementation(
                            test_data, device, precision, impl
                        )
                        
                        # 计算精度指标
                        metrics = self.calculate_accuracy_metrics(
                            cpu_output, device_output, operator_name, f"{device}_{impl}_{precision_name}"
                        )
                        
                        device_results[impl] = {
                            'metrics': metrics,
                            'success': True
                        }
                        print(f"    ✓ 完成")
                        
                    except Exception as e:
                        print(f"    ❌ 失败: {str(e)}")
                        device_results[impl] = {
                            'error': str(e),
                            'status': 'failed'
                        }
                
                precision_results[device] = device_results
            
            results['precision_results'][precision_name] = precision_results
        
        return results
    
    def _generate_summary(
        self, 
        accuracy_results: List[Dict[str, Any]], 
        performance_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """生成测试汇总"""
        
        summary = {
            'total_test_cases': len(accuracy_results),
            'accuracy_summary': {},
            'performance_summary': {}
        }
        
        # 精度汇总
        if accuracy_results:
            precision_types = set()
            for result in accuracy_results:
                precision_types.update(result['precision_results'].keys())
            
            for precision in precision_types:
                precision_summary = {
                    'total_tests': 0,
                    'successful_tests': 0,
                    'avg_cosine_similarity': 0,
                    'avg_rmse': 0
                }
                
                total_cosine = 0
                total_rmse = 0
                success_count = 0
                
                for result in accuracy_results:
                    if precision in result['precision_results']:
                        for device, device_results in result['precision_results'][precision].items():
                            for impl, impl_result in device_results.items():
                                precision_summary['total_tests'] += 1
                                # 检查impl_result是否直接包含success键，或者是嵌套结构
                                if 'success' in impl_result:
                                    # 直接结构（精度测试）
                                    if impl_result['success']:
                                        precision_summary['successful_tests'] += 1
                                        metrics = impl_result['metrics']
                                        total_cosine += metrics.cosine_similarity
                                        total_rmse += metrics.rmse
                                        success_count += 1
                                elif 'full_function' in impl_result:
                                    # 嵌套结构（性能测试）
                                    if impl_result['full_function']['success']:
                                        precision_summary['successful_tests'] += 1
                                        metrics = impl_result['full_function']['metrics']
                                        total_cosine += metrics.cosine_similarity
                                        total_rmse += metrics.rmse
                                        success_count += 1
                
                if success_count > 0:
                    precision_summary['avg_cosine_similarity'] = total_cosine / success_count
                    precision_summary['avg_rmse'] = total_rmse / success_count
                
                summary['accuracy_summary'][precision] = precision_summary
        
        # 性能汇总
        if performance_results:
            precision_types = set()
            for result in performance_results:
                precision_types.update(result['performance_results'].keys())
            
            for precision in precision_types:
                precision_summary = {
                    'total_tests': 0,
                    'successful_tests': 0,
                    'avg_time_ms': 0,
                    'best_implementation': None,
                    'best_time_ms': float('inf')
                }
                
                total_time = 0
                success_count = 0
                
                for result in performance_results:
                    if precision in result['performance_results']:
                        for device, device_results in result['performance_results'][precision].items():
                            for impl, impl_result in device_results.items():
                                precision_summary['total_tests'] += 1
                                # 检查impl_result是否直接包含success键，或者是嵌套结构
                                if 'success' in impl_result:
                                    # 直接结构
                                    if impl_result['success']:
                                        precision_summary['successful_tests'] += 1
                                        metrics = impl_result['metrics']
                                        total_time += metrics.avg_time_ms
                                        success_count += 1
                                        
                                        # 记录最佳实现
                                        if metrics.avg_time_ms < precision_summary['best_time_ms']:
                                            precision_summary['best_time_ms'] = metrics.avg_time_ms
                                            precision_summary['best_implementation'] = f"{device}_{impl}"
                                elif 'full_function' in impl_result:
                                    # 嵌套结构（性能测试）
                                    if impl_result['full_function']['success']:
                                        precision_summary['successful_tests'] += 1
                                        metrics = impl_result['full_function']['metrics']
                                        total_time += metrics.avg_time_ms
                                        success_count += 1
                                        
                                        # 记录最佳实现
                                        if metrics.avg_time_ms < precision_summary['best_time_ms']:
                                            precision_summary['best_time_ms'] = metrics.avg_time_ms
                                            precision_summary['best_implementation'] = f"{device}_{impl}"
                
                if success_count > 0:
                    precision_summary['avg_time_ms'] = total_time / success_count
                
                summary['performance_summary'][precision] = precision_summary
        
        return summary
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """保存测试结果"""
        
        # 转换结果为可序列化格式
        serializable_results = self._make_serializable(results)
        
        # 保存JSON格式
        json_file = self.result_dir / f"{filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 测试结果已保存到: {json_file}")
    
    def _make_serializable(self, obj):
        """将对象转换为可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (AccuracyMetrics, PerformanceMetrics)):
            return obj.__dict__
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)  # 将 PosixPath 转换为字符串
        else:
            return obj
    
    def print_summary_report(self, results: Dict[str, Any]):
        """打印汇总报告"""
        
        print(f"\n{'='*80}")
        print(f"测试汇总报告: {results['operator_name']}")
        print(f"{'='*80}")
        
        summary = results['summary']
        
        print(f"总测试案例数: {summary['total_test_cases']}")
        
        # 精度汇总
        print(f"\n--- 精度测试汇总 ---")
        for precision, precision_summary in summary['accuracy_summary'].items():
            success_rate = precision_summary['successful_tests'] / precision_summary['total_tests'] * 100
            print(f"{precision}:")
            print(f"  成功率: {success_rate:.1f}% ({precision_summary['successful_tests']}/{precision_summary['total_tests']})")
            if precision_summary['successful_tests'] > 0:
                print(f"  平均余弦相似度: {precision_summary['avg_cosine_similarity']:.6f}")
                print(f"  平均RMSE: {precision_summary['avg_rmse']:.6e}")
        
        # 性能汇总
        print(f"\n--- 性能测试汇总 ---")
        for precision, precision_summary in summary['performance_summary'].items():
            success_rate = precision_summary['successful_tests'] / precision_summary['total_tests'] * 100
            print(f"{precision}:")
            print(f"  成功率: {success_rate:.1f}% ({precision_summary['successful_tests']}/{precision_summary['total_tests']})")
            if precision_summary['successful_tests'] > 0:
                print(f"  平均时间: {precision_summary['avg_time_ms']:.3f}ms")
                if precision_summary['best_implementation']:
                    print(f"  最佳实现: {precision_summary['best_implementation']} ({precision_summary['best_time_ms']:.3f}ms)")
    

    
    def list_registered_operators(self):
        """列出已注册的算子"""
        print(f"\n已注册的算子 (共 {len(self.operators)} 个):")
        for name, operator in self.operators.items():
            print(f"  - {name}")
            print(f"    支持精度: {[p.name for p in operator.supported_precisions]}")
            print(f"    支持设备: {[d.value for d in operator.supported_devices]}")