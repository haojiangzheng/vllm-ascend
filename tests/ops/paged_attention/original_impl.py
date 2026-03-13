"""
PagedAttention原始实现
使用传统的attention计算方式
"""

import torch
try:
    import torch_npu
except ImportError:
    torch_npu = None
from typing import Dict, Any


class OriginalPagedAttentionImpl:
    """PagedAttention原始实现"""
    
    def __init__(self):
        self.name = "npu_original"
    
    def prepare_data(self, data: Dict[str, Any], device: str, precision) -> Dict[str, Any]:
        """准备数据用于核心算子执行"""
        # 转换数据类型和设备
        query = data['query'].to(dtype=precision.value, device=device)
        key_cache = data['key_cache'].to(dtype=precision.value, device=device)
        value_cache = data['value_cache'].to(dtype=precision.value, device=device)
        block_table = data['block_table'].to(device=device)
        context_lens = data['context_lens']
        
        # 创建输出tensor
        batch_size = data['metadata']['batch_size']
        num_heads = data['num_heads']
        head_size = data['head_size']
        
        output = torch.empty(
            batch_size, num_heads, head_size,
            dtype=precision.value, device=device
        )
        
        return {
            'query': query,
            'key_cache': key_cache,
            'value_cache': value_cache,
            'block_table': block_table,
            'context_lens': context_lens,
            'output': output,
            'num_kv_heads': data['num_kv_heads'],
            'num_heads': data['num_heads'],
            'scale_value': data['scale']
        }
    
    def execute_core_operator(self, prepared_data: Dict[str, Any]) -> torch.Tensor:
        """执行核心算子 - torch_npu._npu_paged_attention"""
        # 调用NPU核心算子
        torch_npu._npu_paged_attention(
            query=prepared_data['query'],
            key_cache=prepared_data['key_cache'],
            value_cache=prepared_data['value_cache'],
            num_kv_heads=prepared_data['num_kv_heads'],
            num_heads=prepared_data['num_heads'],
            scale_value=prepared_data['scale_value'],
            block_table=prepared_data['block_table'],
            context_lens=prepared_data['context_lens'],
            out=prepared_data['output']
        )
        
        # 返回设备上的原始输出，不进行后处理
        return prepared_data['output']
    
    def run_full_implementation(self, data: Dict[str, Any], device: str, precision) -> torch.Tensor:
        """运行完整实现（包含数据处理和后处理）"""
        # 数据准备
        prepared_data = self.prepare_data(data, device, precision)
        
        # 执行核心算子
        output = self.execute_core_operator(prepared_data)
        
        # 后处理：转换到CPU并转换为float32
        return output.cpu().float()