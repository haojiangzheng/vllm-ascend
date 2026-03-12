"""
GroupGemm算子测试实现 - INT8版本
通用的 GroupGemm 测试，主要测试 num_experts=8 的场景，使用INT8精度
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
from operator_test_framework import PrecisionType
from groupgemm.base_groupgemm import BaseGroupGemmOperatorTest
import numpy as np


class GroupGemmOperatorTest(BaseGroupGemmOperatorTest):
    """GroupGemm INT8算子测试类"""
    
    def __init__(self, num_experts: int = 8, hidden_dim: int = 7168, out_channel: int = 4096, use_nz_format: bool = False):
        super().__init__(
            num_experts=num_experts, 
            hidden_dim=hidden_dim, 
            out_channel=out_channel,
            use_nz_format=use_nz_format
        )
        # INT8版本只支持INT8精度配置
        self.supported_precisions = [PrecisionType.INT8]
    
    def get_precision_config(self) -> Dict[str, Any]:
        """获取INT8精度配置"""
        return {
            'input_dtype': torch.int8,
            'weight_dtype': torch.int8,
            'scale_dtype': torch.bfloat16,
            'per_token_scale_dtype': torch.float32,
            'output_dtype': torch.bfloat16,
            'input_dtype_size': 1,  # INT8 = 1字节
            'weight_dtype_size': 1,
            'output_dtype_size': 2  # BF16 = 2字节
        }
    
    def generate_precision_specific_data(self, seq_len: int, num_experts: int, 
                                       hidden_dim: int, out_channel: int) -> Dict[str, Any]:
        """生成INT8特定的测试数据"""
        # 根据torch_npu.npu_grouped_matmul的量化场景参数定义生成数据
        # x为INT8、weight为INT8、scale为BFLOAT16、per_token_scale为FLOAT32
        x = torch.randint(-128, 127, (seq_len, hidden_dim), dtype=torch.int8)
        weight = torch.randint(-128, 127, (num_experts, hidden_dim, out_channel), dtype=torch.int8)
        scale = torch.randn((num_experts, out_channel), dtype=torch.bfloat16)
        per_token_scale = torch.randn((seq_len,), dtype=torch.float32)
        
        # 量化场景，这些参数为空
        bias = None
        offset = None
        antiquant_scale = None
        antiquant_offset = None

        return {
            'x': x,
            'weight': weight,
            'bias': bias,
            'scale': scale,
            'offset': offset,
            'antiquant_scale': antiquant_scale,
            'antiquant_offset': antiquant_offset,
            'per_token_scale': per_token_scale,
            'seq_len': seq_len,
            'num_experts': num_experts,
            'hidden_dim': hidden_dim,
            'out_channel': out_channel
        }
    
    def get_npu_grouped_matmul_kwargs(self, data: Dict[str, Any], group_list: torch.Tensor) -> Dict[str, Any]:
        """获取INT8版本的npu_grouped_matmul参数"""
        return {
            "x": [data['x']],
            "weight": [data['weight']],
            "scale": [data['scale']],
            "per_token_scale": [data['per_token_scale']],
            "split_item": 2,
            "group_list_type": 1,
            "group_type": 0,
            "group_list": group_list,
            "output_dtype": torch.bfloat16
        }

    
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
        return torch.zeros((seq_len, out_channel), dtype=torch.bfloat16)