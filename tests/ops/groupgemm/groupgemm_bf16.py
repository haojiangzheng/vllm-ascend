"""GroupGemm算子测试实现 - BF16版本
通用的 GroupGemm 测试，主要测试 num_experts=8 的场景，使用BF16精度
根据torch_npu.npu_grouped_matmul的参数定义：
- x为BFLOAT16、weight为BFLOAT16、bias为FLOAT32、output_dtype为BFLOAT16
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from typing import Dict, Any
from operator_test_framework import PrecisionType
from groupgemm.base_groupgemm import BaseGroupGemmOperatorTest


class GroupGemmBF16OperatorTest(BaseGroupGemmOperatorTest):
    """GroupGemm算子测试类 - BF16版本"""
    
    def __init__(self, num_experts: int = 8, hidden_dim: int = 7168, out_channel: int = 4096, use_nz_format: bool = False):
        # 为BF16版本设置特定的算子名称
        operator_name = "GroupGemm_BF16_NZ" if use_nz_format else "GroupGemm_BF16"
        super().__init__(operator_name, num_experts, hidden_dim, out_channel, use_nz_format)
        # BF16版本只支持BF16精度配置
        self.supported_precisions = [PrecisionType.BF16]
    
    def get_precision_config(self) -> Dict[str, Any]:
        """获取BF16精度配置"""
        return {
            'input_dtype': torch.bfloat16,
            'weight_dtype': torch.bfloat16,
            'bias_dtype': torch.float32,
            'output_dtype': torch.bfloat16,
            'input_dtype_size': 2,  # BF16 = 2字节
            'weight_dtype_size': 2,
            'output_dtype_size': 2
        }
    
    def generate_precision_specific_data(self, seq_len: int, num_experts: int, 
                                       hidden_dim: int, out_channel: int) -> Dict[str, Any]:
        """生成BF16特定的测试数据"""
        # 根据torch_npu.npu_grouped_matmul的非量化场景参数定义生成数据
        # x为BFLOAT16、weight为BFLOAT16、bias为FLOAT32
        x = torch.randn((seq_len, hidden_dim), dtype=torch.bfloat16)
        weight = torch.randn((num_experts, hidden_dim, out_channel), dtype=torch.bfloat16)
        bias = torch.randn((num_experts, out_channel), dtype=torch.float32)
        
        # 非量化场景，这些参数为空
        scale = None
        offset = None
        antiquant_scale = None
        antiquant_offset = None
        per_token_scale = None
        
        return {
            'x': x,
            'weight': weight,
            'bias': bias,
            'scale': scale,
            'offset': offset,
            'antiquant_scale': antiquant_scale,
            'antiquant_offset': antiquant_offset,
            'per_token_scale': per_token_scale
        }
    
    def get_npu_grouped_matmul_kwargs(self, data: Dict[str, Any], group_list: torch.Tensor) -> Dict[str, Any]:
        """获取BF16版本的npu_grouped_matmul参数"""
        return {
            "x": [data['x']],
            "weight": [data['weight']],
            "bias": [data['bias']],
            "scale": data['scale'],  # 为空
            "offset": data['offset'],  # 为空
            "antiquant_scale": data['antiquant_scale'],  # 为空
            "antiquant_offset": data['antiquant_offset'],  # 为空
            "split_item": 2,
            "group_list_type": 1,
            "group_type": 0,
            "group_list": group_list,
            "output_dtype": torch.bfloat16
        }


if __name__ == "__main__":
    # 独立测试
    test = GroupGemmBF16OperatorTest(num_experts=8)
    
    # 生成测试数据
    data = test.generate_test_data(seq_len=1024)
    print(f"✅ BF16 GroupGemm测试数据生成完成！")
    print(f"输入形状: {data['x'].shape}, 数据类型: {data['x'].dtype}")
    print(f"权重形状: {data['weight'].shape}, 数据类型: {data['weight'].dtype}")
    print(f"偏置形状: {data['bias'].shape}, 数据类型: {data['bias'].dtype}")