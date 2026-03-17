
"""
FlashAttention算子测试套件
"""

import sys
import os
import torch
import argparse
try:
    import torch_npu
except ImportError:
    pass
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
from tests.base_test_suite import BaseTestSuite
from flashattention.base import FlashAttentionOperatorTest
from operator_test_framework import OperatorTestFramework

class FlashAttentionTestSuite(BaseTestSuite):
    """FlashAttention算子测试套件"""
    
    def __init__(self):
        super().__init__("flash_attention")
        self.operator_test = FlashAttentionOperatorTest()
    
    def register_operator(self):
        """注册FlashAttention算子到测试框架"""
        self.framework.register_operator(self.operator_test)
    
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """创建FlashAttention算子测试案例"""
        return [
            {
                'name': 'standard_bnsd',
                'params': {
                    'batch_size': 4,
                    'num_heads': 32,
                    'seq_len': 128,
                    'head_size': 128,
                    'input_layout': 'BNSD',
                    'sparse_mode': 0
                }
            },
            {
                'name': 'gqa_bnsd',
                'params': {
                    'batch_size': 4,
                    'num_heads': 32,
                    'num_kv_heads': 8,
                    'seq_len': 128,
                    'head_size': 128,
                    'input_layout': 'BNSD',
                    'sparse_mode': 0
                }
            },
            {
                'name': 'causal_mask_bnsd',
                'params': {
                    'batch_size': 4,
                    'num_heads': 32,
                    'seq_len': 128,
                    'head_size': 128,
                    'input_layout': 'BNSD',
                    'sparse_mode': 3 # right down causal
                }
            },
            {
                'name': 'large_seq_bnsd',
                'params': {
                    'batch_size': 1,
                    'num_heads': 16,
                    'seq_len': 2048,
                    'head_size': 128,
                    'input_layout': 'BNSD',
                    'sparse_mode': 0
                }
            },
            {
                'name': 'tnd_layout_basic',
                'params': {
                    'batch_size': 4,
                    'num_heads': 32,
                    'seq_len': 128,
                    'head_size': 128,
                    'input_layout': 'TND',
                    'sparse_mode': 3
                }
            },
            {
                'name': 'tnd_layout_with_block_table',
                'params': {
                    'batch_size': 2,
                    'num_heads': 16,
                    'seq_len': 256,
                    'head_size': 128,
                    'input_layout': 'TND',
                    'sparse_mode': 3,
                    'use_block_table': True,
                    'block_size': 128
                }
            },
            {
                'name': 'tnd_layout_variable_seq_len',
                'params': {
                    'batch_size': 4,
                    'num_heads': 16,
                    'seq_len': 128, # Max seq len
                    'head_size': 128,
                    'input_layout': 'TND',
                    'sparse_mode': 3,
                    'variable_seq_lengths': True,
                    'min_seq_len': 32
                }
            }
        ]
        
    def create_quick_test_cases(self) -> List[Dict[str, Any]]:
        """创建快速测试案例"""
        return [
            {
                'name': 'quick_test_bnsd',
                'params': {
                    'batch_size': 2,
                    'num_heads': 8,
                    'seq_len': 64,
                    'head_size': 64,
                    'input_layout': 'BNSD',
                    'sparse_mode': 0
                }
            }
        ]

    def _select_device(self) -> str:
        npu_available = False
        try:
            import torch_npu
            npu_available = True
        except ImportError:
            pass

        for device_type in self.operator_test.supported_devices:
            if device_type.value == "npu" and npu_available:
                return "npu:0"
        return "unavailable"

    def run_tflops_test(self, precision: str = "fp16", num_warmup: int = 5, num_iterations: int = 10):
        try:
            import matplotlib.pyplot as plt
            import csv
            import time
        except ImportError:
            print("❌ 未找到matplotlib，无法绘制曲线。请安装matplotlib: pip install matplotlib")
            return

        from operator_test_framework import PrecisionType

        precision_map = {
            "fp16": PrecisionType.FP16,
            "bf16": PrecisionType.BF16,
        }
        precision_type = precision_map.get(precision, PrecisionType.FP16)
        device = self._select_device()
        has_npu_runtime = device != "unavailable"
        n_ctx_values = [2 ** i for i in range(10, 15)]
        benchmark_cases = []
        for head_dim in [64, 128]:
            for causal in [True, False]:
                benchmark_cases.append({
                    "batch_size": 4,
                    "num_heads": 32,
                    "head_dim": head_dim,
                    "causal": causal,
                })

        triton_fp16_reference = {
            (64, True): [113.230862, 138.093367, 152.464573, 159.157748, 166.349096],
            (64, False): [141.961500, 161.479554, 162.150925, 165.451044, 166.600750],
            (128, True): [122.107300, 145.557095, 163.869660, 173.262014, 177.130320],
            (128, False): [159.118316, 173.001194, 173.226075, 179.574258, 176.464413],
        }

        os.makedirs("test_results", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        device_tag = device.replace(":", "_")
        csv_file = f"test_results/flashattention_tflops_{precision}_{device_tag}_{timestamp}.csv"

        results = []
        print(f"\n{'='*80}")
        print(f"FlashAttention Forward TFLOPS 测试 ({device}) - Precision: {precision.upper()}")
        print(f"{'='*80}")
        if not has_npu_runtime:
            print("⚠️ 未检测到 torch_npu 或 NPU 设备，将仅绘制 Triton 参考曲线。")
        print(f"{'D':>6} {'Causal':>8} {'N_CTX':>8} {'Provider':>12} {'Time(ms)':>12} {'TFLOPS':>12}")

        measured_curves = {}
        for case in benchmark_cases:
            head_dim = case["head_dim"]
            causal = case["causal"]
            curve_key = (head_dim, causal)
            measured_curves[curve_key] = []

            for n_ctx in n_ctx_values:
                sparse_mode = 3 if causal else 0
                if has_npu_runtime:
                    try:
                        test_data = self.operator_test.generate_test_data(
                            batch_size=case["batch_size"],
                            num_heads=case["num_heads"],
                            seq_len=n_ctx,
                            head_size=head_dim,
                            input_layout="BNSD",
                            sparse_mode=sparse_mode
                        )
                        perf = self.framework.run_core_operator_performance_test_v2(
                            operator_test=self.operator_test,
                            data=test_data,
                            device=device,
                            precision=precision_type,
                            implementation="npu_flash_attention",
                            num_warmup=num_warmup,
                            num_iterations=num_iterations
                        )
                        tflops = self.operator_test.calculate_tflops(test_data, perf.avg_time_ms, mode="fwd") or 0.0
                        measured_curves[curve_key].append(tflops)
                        results.append([head_dim, causal, n_ctx, "optests", perf.avg_time_ms, tflops])
                        print(f"{head_dim:6d} {str(causal):>8} {n_ctx:8d} {'optests':>12} {perf.avg_time_ms:12.3f} {tflops:12.3f}")
                    except Exception as e:
                        measured_curves[curve_key].append(0.0)
                        results.append([head_dim, causal, n_ctx, "optests_failed", -1.0, 0.0])
                        print(f"{head_dim:6d} {str(causal):>8} {n_ctx:8d} {'FAILED':>12} {'-':>12} {0.0:12.3f} ({e})")
                triton_ref_tflops = triton_fp16_reference[(head_dim, causal)][n_ctx_values.index(n_ctx)]
                results.append([head_dim, causal, n_ctx, "triton_fp16_ref", -1.0, triton_ref_tflops])
                print(f"{head_dim:6d} {str(causal):>8} {n_ctx:8d} {'triton_ref':>12} {'-':>12} {triton_ref_tflops:12.3f}")

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["HEAD_DIM", "CAUSAL", "N_CTX", "PROVIDER", "TIME_MS", "TFLOPS"])
            writer.writerows(results)
        print(f"\n✅ 测试结果已保存至 {csv_file}")

        plt.figure(figsize=(14, 6))
        for i, head_dim in enumerate([64, 128], start=1):
            plt.subplot(1, 2, i)
            for causal, color in [(True, "tab:blue"), (False, "tab:orange")]:
                triton_ref = triton_fp16_reference[(head_dim, causal)]
                label_suffix = "causal=True" if causal else "causal=False"
                if has_npu_runtime:
                    measured = measured_curves[(head_dim, causal)]
                    plt.plot(n_ctx_values, measured, marker="o", color=color, linewidth=2, label=f"OpTests {label_suffix}")
                plt.plot(n_ctx_values, triton_ref, marker="x", linestyle="--", color=color, linewidth=1.5, label=f"H100 Triton FP16 {label_suffix}")

            plt.xscale("log", base=2)
            plt.xticks(n_ctx_values, [str(v) for v in n_ctx_values], rotation=20)
            plt.xlabel("N_CTX")
            plt.ylabel("TFLOPS")
            plt.title(f"FlashAttention FWD (B=4, H=32, D={head_dim})")
            plt.grid(True, which="both", ls="-", alpha=0.4)
            plt.legend(fontsize=8)

        plt.tight_layout()
        plot_file = f"test_results/flashattention_tflops_curve_{precision}_{device_tag}_{timestamp}.png"
        plt.savefig(plot_file, dpi=160)
        plt.close()
        print(f"✅ TFLOPS曲线已保存至 {plot_file}")

def main():
    """主函数 - 支持独立运行FlashAttention算子测试"""
    
    parser = argparse.ArgumentParser(description="FlashAttention算子测试")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["accuracy", "performance", "profile", "comprehensive", "tflops"],
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
        "--precision",
        type=str,
        choices=["fp16", "bf16"],
        default="fp16",
        help="tflops模式使用的精度"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="tflops模式预热次数"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="tflops模式测量次数"
    )
    
    args = parser.parse_args()
    
    # 设置测试框架
    framework = OperatorTestFramework(result_dir=args.result_dir)
    
    # 创建并设置FlashAttention测试套件
    suite = FlashAttentionTestSuite()
    suite.setup(framework)
    
    # 根据模式运行测试
    if args.mode == "accuracy":
        suite.run_accuracy_test()
    elif args.mode == "performance":
        suite.run_performance_test()
    elif args.mode == "profile":
        suite.run_profile_test()
    elif args.mode == "comprehensive":
        suite.run_comprehensive_test()
    elif args.mode == "tflops":
        suite.run_tflops_test(
            precision=args.precision,
            num_warmup=args.warmup,
            num_iterations=args.iterations
        )
    
    print(f"\n✓ FlashAttention算子测试完成，结果已保存到 {args.result_dir}")


if __name__ == "__main__":
    main()
