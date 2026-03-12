"""
PagedAttention算子测试套件
"""

import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
from tests.base_test_suite import BaseTestSuite
from paged_attention.base import PagedAttentionOperatorTest


class PagedAttentionTestSuite(BaseTestSuite):
    """PagedAttention算子测试套件"""
    
    def __init__(self):
        super().__init__("paged_attention")
        self.operator_test = PagedAttentionOperatorTest()
    
    def register_operator(self):
        """注册PagedAttention算子到测试框架"""
        self.framework.register_operator(self.operator_test)
    
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """创建PagedAttention算子测试案例"""
        return [
            # qwen3-32b tp8
            {
                'name': 'qwen3-32b_tp8_1k',
                'params': {
                    'batch_size': 128,
                    'num_heads': 8,
                    'num_kv_heads': 1,
                    'head_size': 128,
                    'max_seq_len': 1024,
                    'num_blocks': 10000,
                    'block_size': 128
                }
            },
            {
                'name': 'qwen3-32b_tp8_10k',
                'params': {
                    'batch_size': 128,
                    'num_heads': 8,
                    'num_kv_heads': 1,
                    'head_size': 128,
                    'max_seq_len': 10240,
                    'num_blocks': 10000,
                    'block_size': 128
                }
            },
            {
                'name': 'qwen3-32b_tp8_30k',
                'params': {
                    'batch_size': 128,
                    'num_heads': 8,
                    'num_kv_heads': 1,
                    'head_size': 128,
                    'max_seq_len': 32768,
                    'num_blocks': 10000,
                    'block_size': 128
                }
            },
            {
                'name': 'qwen3-235b-tp4dpx_8k',
                'params': {
                    'batch_size': 12,
                    'num_heads': 16,
                    'num_kv_heads': 1,
                    'head_size': 128,
                    'max_seq_len': 8192,
                    'num_blocks': 10000,
                    'block_size': 128
                }
            },
            {
                'name': 'qwen3-235b-tp1dpx_8k',
                'params': {
                    'batch_size': 12,
                    'num_heads': 64,
                    'num_kv_heads': 4,
                    'head_size': 128,
                    'max_seq_len': 8192,
                    'num_blocks': 10000,
                    'block_size': 128
                }
            },
            {
                'name': 'qwen2.5 72b-tp8_4k',
                'params': {
                    'batch_size': 256,
                    'num_heads': 8,
                    'num_kv_heads': 1,
                    'head_size': 128,
                    'max_seq_len': 4096,
                    'num_blocks': 10000,
                    'block_size': 128
                }
            }
        ]
    
    def create_full_test_cases(self, num_cases: int = 50) -> List[Dict[str, Any]]:
        """创建全面测试案例，使用随机参数生成
        
        参数范围：
        - batch_size: 1-512
        - num_heads: [8, 16, 64]
        - num_kv_heads: [1, 4]
        - head_size: 128 (固定)
        - max_seq_len: 1-65536 (64k)
        - num_blocks: 9000-13000
        - block_size: 128 (固定)
        """
        test_cases = []
        
        # 可选的num_heads和num_kv_heads值
        num_heads_options = [8, 16, 64]
        num_kv_heads_options = [1, 4]
        
        for i in range(num_cases):
            # 随机生成参数
            batch_size = random.randint(1, 512)
            num_heads = random.choice(num_heads_options)
            num_kv_heads = random.choice(num_kv_heads_options)
            head_size = 128  # 固定值
            max_seq_len = random.randint(1, 65536)  # 1到64k
            num_blocks = random.randint(9000, 13000)
            block_size = 128  # 固定值
            
            test_case = {
                'name': f'full_test_case_{i+1}',
                'params': {
                    'batch_size': batch_size,
                    'num_heads': num_heads,
                    'num_kv_heads': num_kv_heads,
                    'head_size': head_size,
                    'max_seq_len': max_seq_len,
                    'num_blocks': num_blocks,
                    'block_size': block_size
                }
            }
            test_cases.append(test_case)
        
        return test_cases
    
    def create_quick_test_cases(self) -> List[Dict[str, Any]]:
        """创建快速测试案例"""
        return [
            {
                'name': 'quick_medium',
                'params': {
                    'batch_size': 128,
                    'num_heads': 8,
                    'num_kv_heads': 1,
                    'head_size': 128,
                    'max_seq_len': 4096,
                    'num_blocks': 10000,
                    'block_size': 128
                }
            }
        ]
    
    def print_test_cases_summary(self, test_cases: List[Dict[str, Any]]):
        """打印测试案例的参数统计摘要"""
        if not test_cases:
            return
        
        print(f"\n📊 测试案例参数统计 (共 {len(test_cases)} 个案例):")
        print("-" * 60)
        
        # 收集所有参数值
        batch_sizes = [case['params']['batch_size'] for case in test_cases]
        num_heads_list = [case['params']['num_heads'] for case in test_cases]
        num_kv_heads_list = [case['params']['num_kv_heads'] for case in test_cases]
        max_seq_lens = [case['params']['max_seq_len'] for case in test_cases]
        num_blocks_list = [case['params']['num_blocks'] for case in test_cases]
        
        print(f"batch_size    : {min(batch_sizes):4d} - {max(batch_sizes):4d}")
        print(f"num_heads     : {sorted(set(num_heads_list))}")
        print(f"num_kv_heads  : {sorted(set(num_kv_heads_list))}")
        print(f"head_size     : 128 (固定)")
        print(f"max_seq_len   : {min(max_seq_lens):5d} - {max(max_seq_lens):5d}")
        print(f"num_blocks    : {min(num_blocks_list):5d} - {max(num_blocks_list):5d}")
        print(f"block_size    : 128 (固定)")
        print("-" * 60)
    
    def create_profile_test_cases(self) -> List[Dict[str, Any]]:
        """创建专门的 Profile 测试案例"""
        return [
            {
                'name': 'qwen3-32b_tp8_10k_profile',
                'params': {
                    'batch_size': 128,
                    'num_heads': 8,
                    'num_kv_heads': 1,
                    'head_size': 128,
                    'max_seq_len': 10019,
                    'num_blocks': 9695,
                    'block_size': 128
                }
            },
            {
                'name': 'qwen3-32b_tp8_30k_profile',
                'params': {
                    'batch_size': 128,
                    'num_heads': 8,
                    'num_kv_heads': 1,
                    'head_size': 128,
                    'max_seq_len': 32768,
                    'num_blocks': 9695,
                    'block_size': 128
                }
            }
        ]

def main():
    """主函数 - 支持独立运行PagedAttention算子测试"""
    import argparse
    import sys
    import os
    
    # 添加父目录到Python路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from operator_test_framework import OperatorTestFramework
    
    parser = argparse.ArgumentParser(description="PagedAttention算子测试")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["accuracy", "performance", "profile", "comprehensive", "fulltest"],
        default="comprehensive",
        help="测试模式 (默认: comprehensive)"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="test_results",
        help="测试结果保存目录 (默认: test_results)"
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=50,
        help="fulltest模式下生成的随机测试案例数量 (默认: 50)"
    )
    
    args = parser.parse_args()
    
    # 设置测试框架
    framework = OperatorTestFramework(result_dir=args.result_dir)
    
    # 创建并设置PagedAttention测试套件
    pa_suite = PagedAttentionTestSuite()
    pa_suite.setup(framework)
    
    # 根据模式运行测试
    if args.mode == "accuracy":
        results = pa_suite.run_accuracy_test()
    elif args.mode == "performance":
        results = pa_suite.run_performance_test()
    elif args.mode == "profile":
        results = pa_suite.run_profile_test()
    elif args.mode == "comprehensive":
        results = pa_suite.run_comprehensive_test()
    elif args.mode == "fulltest":
        # 生成随机测试案例并仅运行精度测试
        full_test_cases = pa_suite.create_full_test_cases(num_cases=args.num_cases)
        print(f"\n🎯 运行fulltest模式，生成 {len(full_test_cases)} 个随机测试案例")
        pa_suite.print_test_cases_summary(full_test_cases)
        results = pa_suite.run_accuracy_test(full_test_cases)
    
    print(f"\n✓ PagedAttention算子测试完成，结果已保存到 {args.result_dir}")


if __name__ == "__main__":
    main()