"""
FlashAttention算子测试模块

包含不同实现的FlashAttention算子：
- npu_flash_attention: 使用 npu_fused_infer_attention_score 的实现
"""

from .base import FlashAttentionOperatorTest
from .impl import FlashAttentionNpuImpl

__all__ = [
    'FlashAttentionOperatorTest',
    'FlashAttentionNpuImpl'
]
