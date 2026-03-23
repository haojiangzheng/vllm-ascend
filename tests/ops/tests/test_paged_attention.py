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

    def run_latency_plot_test(
        self,
        seqlen_start: int = 1024,
        seqlen_end: int = 32768,
        seqlen_step: int = 1024,
        batch_min: int = 1,
        batch_max: int = 128,
        num_warmup: int = 5,
        num_iterations: int = 20
    ):
        try:
            import matplotlib.pyplot as plt
            import csv
            import time
        except ImportError:
            print("❌ 未找到matplotlib，无法绘制曲线。请安装matplotlib: pip install matplotlib")
            return

        from operator_test_framework import PrecisionType

        device = "npu:0"
        precision_type = PrecisionType.BF16
        implementations = self.operator_test.get_available_implementations(device)
        if not implementations:
            print("❌ 未找到可用的 PagedAttention 实现")
            return

        os.makedirs("test_results", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        seqlens = list(range(seqlen_start, seqlen_end + 1, seqlen_step))
        if seqlens[-1] != seqlen_end:
            seqlens.append(seqlen_end)

        print(f"\n{'='*80}")
        print("PagedAttention 延迟画图测试")
        print(f"设备: {device}, 实现: {', '.join(implementations)}, 精度: {precision_type.name}")
        print(f"{'='*80}")

        seqlen_latency_ms = {impl: [] for impl in implementations}
        seqlen_rows = []

        print("\n📈 曲线1: batch=128, seqlen=1024..32768")
        for impl in implementations:
            print(f"\n  实现: {impl}")
            for seq_len in seqlens:
                try:
                    test_data = self.operator_test.generate_test_data(
                        batch_size=128,
                        num_heads=8,
                        num_kv_heads=1,
                        head_size=128,
                        max_seq_len=seq_len,
                        num_blocks=10000,
                        block_size=128,
                    )
                    result = self.framework.run_core_operator_performance_test_v2(
                        operator_test=self.operator_test,
                        data=test_data,
                        device=device,
                        precision=precision_type,
                        implementation=impl,
                        num_warmup=num_warmup,
                        num_iterations=num_iterations
                    )
                    latency = result.avg_time_ms
                except Exception as e:
                    print(f"  ❌ {impl} seqlen={seq_len} 测试失败: {e}")
                    latency = 0.0
                seqlen_latency_ms[impl].append(latency)
                seqlen_rows.append([impl, seq_len, latency])
                print(f"  {impl} seqlen={seq_len:5d}, latency={latency:.4f} ms")

        batch_sizes = list(range(batch_min, batch_max + 1))
        batch_curve_seq_lens = [10000, 30000]
        batch_latency_ms = {
            impl: {fixed_seq: [] for fixed_seq in batch_curve_seq_lens}
            for impl in implementations
        }
        batch_rows = []

        print("\n📈 曲线2: batch=1..128, seqlen固定10k/30k")
        for impl in implementations:
            print(f"\n  实现: {impl}")
            for fixed_seq in batch_curve_seq_lens:
                print(f"\n  固定 seqlen={fixed_seq}")
                for batch_size in batch_sizes:
                    try:
                        test_data = self.operator_test.generate_test_data(
                            batch_size=batch_size,
                            num_heads=8,
                            num_kv_heads=1,
                            head_size=128,
                            max_seq_len=fixed_seq,
                            num_blocks=10000,
                            block_size=128,
                        )
                        result = self.framework.run_core_operator_performance_test_v2(
                            operator_test=self.operator_test,
                            data=test_data,
                            device=device,
                            precision=precision_type,
                            implementation=impl,
                            num_warmup=num_warmup,
                            num_iterations=num_iterations
                        )
                        latency = result.avg_time_ms
                    except Exception as e:
                        print(f"    ❌ {impl} batch={batch_size} 测试失败: {e}")
                        latency = 0.0
                    batch_latency_ms[impl][fixed_seq].append(latency)
                    batch_rows.append([impl, fixed_seq, batch_size, latency])
                    print(f"    {impl} batch={batch_size:3d}, latency={latency:.4f} ms")

        seqlen_csv = f"test_results/paged_attention_latency_vs_seqlen_{device.replace(':', '_')}_{timestamp}.csv"
        batch_csv = f"test_results/paged_attention_latency_vs_batch_{device.replace(':', '_')}_{timestamp}.csv"
        seqlen_plot = f"test_results/paged_attention_latency_vs_seqlen_{device.replace(':', '_')}_{timestamp}.png"
        batch_plot = f"test_results/paged_attention_latency_vs_batch_{device.replace(':', '_')}_{timestamp}.png"

        try:
            with open(seqlen_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['implementation', 'seq_len', 'latency_ms'])
                writer.writerows(seqlen_rows)
            print(f"\n✅ seqlen曲线数据已保存: {seqlen_csv}")
        except Exception as e:
            print(f"❌ 保存seqlen CSV失败: {e}")

        try:
            with open(batch_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['implementation', 'seq_len_fixed', 'batch_size', 'latency_ms'])
                writer.writerows(batch_rows)
            print(f"✅ batch曲线数据已保存: {batch_csv}")
        except Exception as e:
            print(f"❌ 保存batch CSV失败: {e}")

        try:
            plt.figure(figsize=(12, 7))
            markers = ['o', 's', '^', 'd', 'x', '*']
            for idx, impl in enumerate(implementations):
                marker = markers[idx % len(markers)]
                plt.plot(seqlens, seqlen_latency_ms[impl], marker=marker, linewidth=2, markersize=4, label=impl)
            plt.xlabel('Sequence Length')
            plt.ylabel('Latency (ms)')
            plt.title(f'PagedAttention Latency vs Sequence Length (batch=128, {device}, all implementations)')
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.legend()
            plt.savefig(seqlen_plot)
            plt.close()
            print(f"✅ seqlen曲线图已保存: {seqlen_plot}")
        except Exception as e:
            print(f"❌ 绘制seqlen曲线失败: {e}")

        try:
            plt.figure(figsize=(12, 7))
            markers = ['o', 's', '^', 'd', 'x', '*']
            for idx, impl in enumerate(implementations):
                marker = markers[idx % len(markers)]
                plt.plot(
                    batch_sizes,
                    batch_latency_ms[impl][10000],
                    marker=marker,
                    linewidth=2,
                    markersize=4,
                    label=f'{impl}, seqlen=10k'
                )
                plt.plot(
                    batch_sizes,
                    batch_latency_ms[impl][30000],
                    marker=marker,
                    linestyle='--',
                    linewidth=2,
                    markersize=4,
                    label=f'{impl}, seqlen=30k'
                )
            plt.xlabel('Batch Size')
            plt.ylabel('Latency (ms)')
            plt.title(f'PagedAttention Latency vs Batch Size ({device}, all implementations)')
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.legend()
            plt.savefig(batch_plot)
            plt.close()
            print(f"✅ batch曲线图已保存: {batch_plot}")
        except Exception as e:
            print(f"❌ 绘制batch曲线失败: {e}")

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
        choices=["accuracy", "performance", "profile", "comprehensive", "fulltest", "latency"],
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
    elif args.mode == "latency":
        results = pa_suite.run_latency_plot_test()
    
    print(f"\n✓ PagedAttention算子测试完成，结果已保存到 {args.result_dir}")


if __name__ == "__main__":
    main()
