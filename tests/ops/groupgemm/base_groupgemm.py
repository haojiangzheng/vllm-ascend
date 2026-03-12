"""
GroupGemm算子基础抽象类
提供INT8和BF16版本的通用实现
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
try:
    import torch_npu
except ImportError:
    torch_npu = None
import csv
from typing import Dict, Any, List, Optional, Tuple
from operator_test_framework import BaseOperatorTest, PrecisionType
import numpy as np
from abc import ABC, abstractmethod


class BaseGroupGemmOperatorTest(BaseOperatorTest, ABC):
    """GroupGemm算子基础抽象类"""
    
    def __init__(self, operator_name: str = None, num_experts: int = 8, hidden_dim: int = 7168, 
                 out_channel: int = 4096, use_nz_format: bool = False):
        # 如果没有指定operator_name，根据是否使用NZ格式自动设置
        if operator_name is None:
            operator_name = "GroupGemm_NZ" if use_nz_format else "GroupGemm"
        
        super().__init__(operator_name)
        self.num_experts = num_experts      # 专家数量，默认8
        self.hidden_dim = hidden_dim        # 隐藏维度
        self.out_channel = out_channel      # 输出通道
        self.use_nz_format = use_nz_format  # 是否使用NZ格式（格式29）
        
    @abstractmethod
    def get_precision_config(self) -> Dict[str, Any]:
        """获取精度配置
        
        Returns:
            Dict[str, Any]: 精度配置字典，包含数据类型和输出类型
        """
        pass
    
    @abstractmethod
    def generate_precision_specific_data(self, seq_len: int, num_experts: int, 
                                       hidden_dim: int, out_channel: int) -> Dict[str, Any]:
        """生成精度特定的测试数据
        
        Args:
            seq_len: 序列长度
            num_experts: 专家数量
            hidden_dim: 隐藏维度
            out_channel: 输出通道
            
        Returns:
            Dict[str, Any]: 精度特定的数据
        """
        pass
    
    @abstractmethod
    def get_npu_grouped_matmul_kwargs(self, data: Dict[str, Any], group_list: torch.Tensor) -> Dict[str, Any]:
        """获取npu_grouped_matmul的参数
        
        Args:
            data: 测试数据
            group_list: 分组列表
            
        Returns:
            Dict[str, Any]: npu_grouped_matmul的参数字典
        """
        pass
    
    def _apply_nz_format(self, weight: torch.Tensor) -> torch.Tensor:
        """应用NZ格式（格式29）到权重张量
        
        Args:
            weight: 权重张量，必须在NPU设备上
            
        Returns:
            torch.Tensor: NZ格式的权重张量
        """
        try:
            # 确保张量在NPU设备上
            if not weight.device.type == 'npu':
                raise ValueError(f"NZ格式转换需要在NPU设备上进行，当前设备: {weight.device}")
            
            # 使用torch_npu.npu_format_cast将权重转换为NZ格式（格式29）
            nz_weight = torch_npu.npu_format_cast(weight, 29)
            return nz_weight
        except Exception as e:
            print(f"警告：NZ格式转换失败，使用原始格式: {e}")
            return weight
    
    def generate_test_data(self, seq_len: int = 1024, num_experts: int = None, 
                          hidden_dim: int = None, out_channel: int = None, **kwargs) -> Dict[str, Any]:
        """生成测试数据
        
        Args:
            seq_len: 序列长度
            num_experts: 专家数量，如果不指定则使用默认值
            hidden_dim: 隐藏维度，如果不指定则使用默认值
            out_channel: 输出通道，如果不指定则使用默认值
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 测试数据字典
        """
        if num_experts is None:
            num_experts = self.num_experts
        if hidden_dim is None:
            hidden_dim = self.hidden_dim
        if out_channel is None:
            out_channel = self.out_channel
            
        # 计算分组信息（通用逻辑）
        base = seq_len // num_experts
        remainder = seq_len % num_experts
        
        group_item_array = [base] * num_experts
        for i in range(remainder):
            group_item_array[i] += 1
            
        group_list = torch.tensor(group_item_array, dtype=torch.int64)
        
        # 获取精度特定的数据
        precision_data = self.generate_precision_specific_data(seq_len, num_experts, hidden_dim, out_channel)
        
        # 合并通用数据和精度特定数据
        result = {
            'group_list': group_list,
            'seq_len': seq_len,
            'num_experts': num_experts,
            'hidden_dim': hidden_dim,
            'out_channel': out_channel,
            'use_nz_format': self.use_nz_format
        }
        result.update(precision_data)
        
        return result
    
    def run_cpu_reference(self, data: Dict[str, Any]) -> torch.Tensor:
        """CPU参考实现（简化版本）
        
        Args:
            data: 测试数据
            
        Returns:
            torch.Tensor: CPU计算结果
        """
        # 对于 groupgemm，CPU参考实现比较复杂
        # 这里返回一个占位符结果
        seq_len = data['seq_len']
        out_channel = data.get('out_channel', self.out_channel)
        precision_config = self.get_precision_config()
        output_dtype = precision_config.get('output_dtype', torch.bfloat16)
        return torch.zeros((seq_len, out_channel), dtype=output_dtype)
    
    def run_device_implementation(
        self, 
        data: Dict[str, Any], 
        device: str, 
        precision: PrecisionType,
        implementation: str = "default"
    ) -> torch.Tensor:
        """设备实现
        
        Args:
            data: 测试数据
            device: 设备类型
            precision: 精度类型
            implementation: 实现方式
            
        Returns:
            torch.Tensor: 设备计算结果
        """
        if not device.startswith("npu"):
            raise ValueError("GroupGemm 目前只支持 NPU 设备")
        
        # 将数据移动到设备（通用逻辑）
        device_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                device_data[key] = value.to(device)
            else:
                device_data[key] = value
        
        # 获取npu_grouped_matmul的参数
        kwargs = self.get_npu_grouped_matmul_kwargs(device_data, device_data['group_list'])
        
        # 执行 groupgemm
        result = torch_npu.npu_grouped_matmul(**kwargs)
        return result
    
    def run_core_operator(
        self, 
        data: Dict[str, Any], 
        device: str, 
        precision: PrecisionType,
        implementation: str = "default"
    ) -> torch.Tensor:
        """运行核心算子操作（用于精确性能测试）
        
        Args:
            data: 测试数据
            device: 设备类型
            precision: 精度类型
            implementation: 实现方式
            
        Returns:
            torch.Tensor: 计算结果
        """
        if not device.startswith("npu"):
            raise ValueError("GroupGemm 目前只支持 NPU 设备")
        
        # 获取npu_grouped_matmul的参数（数据已经在设备上）
        kwargs = self.get_npu_grouped_matmul_kwargs(data, data['group_list'])
        
        # 只执行核心算子操作
        result = torch_npu.npu_grouped_matmul(**kwargs)
        return result
    
    def _prepare_data_for_core_operator(
        self, 
        data: Dict[str, Any], 
        device: str, 
        precision: PrecisionType,
        implementation: str = "default"
    ) -> Dict[str, Any]:
        """为核心算子准备数据（数据预处理，不计入性能测试时间）
        
        Args:
            data: 原始测试数据
            device: 设备类型
            precision: 精度类型
            implementation: 实现方式
            
        Returns:
            Dict[str, Any]: 准备好的数据，包含 npu_grouped_matmul 的所有参数
        """
        if not device.startswith("npu"):
            raise ValueError("GroupGemm 目前只支持 NPU 设备")
        
        # 将数据移动到设备
        device_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                device_data[key] = value.to(device)
            else:
                device_data[key] = value
        
        # 如果使用NZ格式，对权重进行格式转换
        if self.use_nz_format and 'weight' in device_data:
            device_data['weight'] = self._apply_nz_format(device_data['weight'])
        
        # 预先计算 npu_grouped_matmul 的参数
        kwargs = self.get_npu_grouped_matmul_kwargs(device_data, device_data['group_list'])
        
        return kwargs
    
    def _execute_core_operator(
        self, 
        prepared_data: Dict[str, Any], 
        implementation: str = "default"
    ) -> torch.Tensor:
        """执行核心算子操作（只计算核心算子执行时间）
        
        Args:
            prepared_data: 预处理好的数据（来自 _prepare_data_for_core_operator）
            implementation: 实现方式
            
        Returns:
            torch.Tensor: 计算结果
        """
        # 直接执行核心算子，不包含任何数据预处理
        result = torch_npu.npu_grouped_matmul(**prepared_data)
        return result

    def get_available_implementations(self, device: str) -> List[str]:
        """获取可用的实现方式
        
        Args:
            device: 设备类型
            
        Returns:
            List[str]: 可用实现列表
        """
        if device.startswith("npu"):
            return ["npu_grouped_matmul"]
        else:
            return []
    
    def calculate_flops(self, data: Dict[str, Any]) -> Optional[float]:
        """计算FLOPS
        
        Args:
            data: 测试数据
            
        Returns:
            Optional[float]: FLOPS数量
        """
        seq_len = data.get('seq_len', 0)
        hidden_dim = data.get('hidden_dim', 0)
        out_channel = data.get('out_channel', 0)
        num_experts = data.get('num_experts', 0)
        
        if seq_len <= 0 or hidden_dim <= 0 or out_channel <= 0 or num_experts <= 0:
            return None
        
        # GroupGemm的FLOPS计算：总的矩阵乘法运算量
        # 每个token都要与对应expert的权重相乘
        # 矩阵乘法: seq_len * hidden_dim * out_channel * 2 (乘法+加法)
        # 注意：这里seq_len是总的序列长度，每个token只与一个expert计算
        total_flops = seq_len * hidden_dim * out_channel * 2
        
        return float(total_flops)

    def calculate_throughput(self, data: Dict[str, Any], avg_time_ms: float) -> Optional[float]:
        """计算吞吐量（GOPS - 每秒十亿次操作）
        
        Args:
            data: 测试数据
            avg_time_ms: 平均执行时间（毫秒）
            
        Returns:
            Optional[float]: GOPS值
        """
        if avg_time_ms <= 0:
            return None
            
        flops = self.calculate_flops(data)
        if flops is None:
            return None
            
        # 转换为GOPS：FLOPS / (时间_秒 * 10^9)
        avg_time_s = avg_time_ms / 1000.0
        gops = flops / (avg_time_s * 1e9)
        
        return gops

    def calculate_bandwidth(self, data: Dict[str, Any], avg_time_ms: float) -> Optional[float]:
        """计算内存带宽（GB/s）
        
        Args:
            data: 测试数据
            avg_time_ms: 平均执行时间（毫秒）
            
        Returns:
            Optional[float]: 内存带宽（GB/s）
        """
        if avg_time_ms <= 0:
            return None
            
        # 计算总内存访问量（字节）
        seq_len = data.get('seq_len', 0)
        hidden_dim = data.get('hidden_dim', 0)
        out_channel = data.get('out_channel', 0)
        num_experts = data.get('num_experts', 0)
        
        if seq_len <= 0 or hidden_dim <= 0 or out_channel <= 0 or num_experts <= 0:
            return None
            
        precision_config = self.get_precision_config()
        input_dtype_size = precision_config.get('input_dtype_size', 2)  # BF16默认2字节
        weight_dtype_size = precision_config.get('weight_dtype_size', 2)
        output_dtype_size = precision_config.get('output_dtype_size', 2)
        
        # 计算内存访问量（字节）
        # 输入读取：seq_len * hidden_dim * input_dtype_size
        # 权重读取：num_experts * hidden_dim * out_channel * weight_dtype_size
        # 输出写入：seq_len * out_channel * output_dtype_size
        input_bytes = seq_len * hidden_dim * input_dtype_size
        weight_bytes = num_experts * hidden_dim * out_channel * weight_dtype_size
        output_bytes = seq_len * out_channel * output_dtype_size
        
        # 如果有bias，也要计算bias的内存访问
        if data.get('bias') is not None:
            bias_dtype_size = precision_config.get('bias_dtype_size', 4)  # FP32默认4字节
            bias_bytes = num_experts * out_channel * bias_dtype_size
        else:
            bias_bytes = 0
            
        # 如果有scale等量化参数，也要计算
        scale_bytes = 0
        if data.get('scale') is not None:
            scale_bytes += num_experts * out_channel * 2  # BF16
        if data.get('per_token_scale') is not None:
            scale_bytes += seq_len * 4  # FP32
            
        total_bytes = input_bytes + weight_bytes + output_bytes + bias_bytes + scale_bytes
        
        # 转换为GB/s：总字节数 / (时间_秒 * 10^9)
        avg_time_s = avg_time_ms / 1000.0
        bandwidth_gb_s = total_bytes / (avg_time_s * 1e9)
        
        return bandwidth_gb_s


    
    def calculate_memory_usage(self, data: Dict[str, Any]) -> Dict[str, float]:
        """计算内存使用量（详细版本）
        
        Args:
            data: 测试数据
            
        Returns:
            Dict[str, float]: 内存使用量信息（单位：MB）
        """
        seq_len = data.get('seq_len', 0)
        hidden_dim = data.get('hidden_dim', 0)
        out_channel = data.get('out_channel', 0)
        num_experts = data.get('num_experts', 0)
        
        precision_config = self.get_precision_config()
        input_dtype_size = precision_config.get('input_dtype_size', 2)  # BF16默认2字节
        weight_dtype_size = precision_config.get('weight_dtype_size', 2)
        output_dtype_size = precision_config.get('output_dtype_size', 2)
        
        # 计算各部分内存使用（字节）
        input_memory = seq_len * hidden_dim * input_dtype_size
        weight_memory = num_experts * hidden_dim * out_channel * weight_dtype_size
        output_memory = seq_len * out_channel * output_dtype_size
        
        # 转换为MB
        input_mb = input_memory / (1024 * 1024)
        weight_mb = weight_memory / (1024 * 1024)
        output_mb = output_memory / (1024 * 1024)
        total_mb = input_mb + weight_mb + output_mb
        
        return {
            'input_memory_mb': input_mb,
            'weight_memory_mb': weight_mb,
            'output_memory_mb': output_mb,
            'total_memory_mb': total_mb
        }
    
    def validate_result(self, result: torch.Tensor, data: Dict[str, Any]) -> bool:
        """验证结果的有效性
        
        Args:
            result: 计算结果
            data: 测试数据
            
        Returns:
            bool: 验证是否通过
        """
        if result is None:
            return False
        
        expected_shape = (data['seq_len'], data.get('out_channel', self.out_channel))
        if result.shape != expected_shape:
            return False
        
        # 检查是否有NaN或Inf
        if torch.isnan(result).any() or torch.isinf(result).any():
            return False
        
        return True