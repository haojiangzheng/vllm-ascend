"""
RMSNorm算子测试套件
"""

import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
from tests.base_test_suite import BaseTestSuite
from rmsnorm.rmsnorm_operator import RMSNormOperatorTest
from operator_test_framework import OperatorTestFramework

class RMSNormTestSuite(BaseTestSuite):
    """RMSNorm算子测试套件"""
    
    def __init__(self):
        super().__init__("RMSNorm")
        self.operator_test = RMSNormOperatorTest()
    
    def register_operator(self):
        """注册RMSNorm算子到测试框架"""
        if self.framework:
            self.framework.register_operator(self.operator_test)
    
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """创建RMSNorm算子测试案例"""
        return [
            {
                'name': 'standard_bert_base',
                'params': {
                    'shape': (32, 128, 768), # Batch, Seq, Hidden
                    'eps': 1e-6
                }
            },
            {
                'name': 'standard_llama_7b',
                'params': {
                    'shape': (1, 2048, 4096), # Batch, Seq, Hidden
                    'eps': 1e-5
                }
            },
             {
                'name': 'large_hidden',
                'params': {
                    'shape': (8, 512, 8192), 
                    'eps': 1e-6
                }
            }
        ]
        
    def create_quick_test_cases(self) -> List[Dict[str, Any]]:
        """创建快速测试案例"""
        return [
             {
                'name': 'quick_small',
                'params': {
                    'shape': (4, 32, 128),
                    'eps': 1e-6
                }
            }
        ]

    def run_performance_test(self, test_cases: List[Dict[str, Any]] = None, include_core_analysis: bool = False, precision_type: Any = None):
        """运行性能测试 - 覆盖基类方法以支持CPU，并使用V2测试"""
        if test_cases is None:
            test_cases = self.create_test_cases()
            
        print(f"\n{'='*80}")
        print(f"{self.operator_name}算子核心性能测试 (V2)")
        print(f"{'='*80}")
        
        # Check supported devices
        devices = []
        from operator_test_framework import DeviceType
        for dt in self.operator_test.supported_devices:
             if dt == DeviceType.CPU:
                 devices.append("cpu")
             elif dt == DeviceType.NPU:
                 devices.append("npu:0")
        
        for test_case in test_cases:
            test_name = test_case['name']
            params = test_case['params']
            print(f"\n🎯 测试用例: {test_name}")
            
            data = self.operator_test.generate_test_data(**params)
            
            for device in devices:
                implementations = self.operator_test.get_available_implementations(device)
                for impl in implementations:
                    for precision in self.operator_test.supported_precisions:
                        print(f"  运行 {device} - {precision.name} - {impl} ...")
                        try:
                            # Use run_core_operator_performance_test_v2 from framework
                            metrics = self.framework.run_core_operator_performance_test_v2(
                                self.operator_test, data, device, precision, impl, num_warmup=10, num_iterations=50
                            )
                            print(f"    平均时间: {metrics.avg_time_ms:.3f} ms")
                            if metrics.throughput:
                                print(f"    吞吐量: {metrics.throughput:.2f} GFLOPS")
                            if metrics.bandwidth_gb_s:
                                print(f"    带宽: {metrics.bandwidth_gb_s:.2f} GB/s")
                                
                        except Exception as e:
                            print(f"    ❌ 失败: {e}")
                            import traceback
                            traceback.print_exc()

    def run_bandwidth_test(self):
        """运行带宽测试并绘制曲线"""
        try:
            import matplotlib.pyplot as plt
            import csv
            import time
        except ImportError:
            print("❌ 未找到matplotlib，无法绘制曲线。请安装matplotlib: pip install matplotlib")
            return

        import torch
        try:
            import torch_npu
        except ImportError:
            pass
        
        from operator_test_framework import PrecisionType, DeviceType

        # 确定设备
        device = None
        npu_available = False
        try:
            import torch_npu
            npu_available = True
        except ImportError:
            pass

        for dt in self.operator_test.supported_devices:
             if dt == DeviceType.NPU and npu_available:
                 device = "npu:0"
                 break
        
        if device is None:
             # Fallback to cuda if available
             if torch.cuda.is_available():
                 device = "cuda:0"
             elif torch.backends.mps.is_available():
                 device = "mps"
             else:
                 device = "cpu"

        print(f"\n{'='*80}")
        print(f"RMSNorm算子带宽测试 ({device})")
        print(f"{'='*80}")
        
        # Ensure test_results directory exists
        os.makedirs("test_results", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Test 1: Varying Total Size (Fixed Hidden=4096)
        print("\n--- 测试 1: Varying Total Size (Fixed Hidden=4096) ---")
        sizes = [2**i for i in range(12, 28)] # 4K to 134M elements
        hidden_size = 4096
        bandwidths_size = []
        results_size = []
        
        for size in sizes:
            # Ensure size is at least hidden_size and divisible
            if size < hidden_size:
                effective_size = hidden_size
            else:
                effective_size = (size // hidden_size) * hidden_size
            
            rows = effective_size // hidden_size
            shape = (rows, hidden_size)
            
            print(f"测试大小: {effective_size} (Shape: {shape})")
            try:
                test_data = self.operator_test.generate_test_data(shape=shape)
                
                result = self.framework.run_core_operator_performance_test_v2(
                    operator_test=self.operator_test,
                    data=test_data,
                    device=device,
                    precision=PrecisionType.BF16, 
                    implementation="default",
                    num_warmup=10,
                    num_iterations=50
                )
                
                bw = result.bandwidth_gb_s
                if bw is None:
                    bw = 0.0
                
                bandwidths_size.append(bw)
                results_size.append([effective_size, hidden_size, bw])
                print(f"  带宽: {bw:.4f} GB/s")
                
            except Exception as e:
                print(f"  测试失败: {e}")
                bandwidths_size.append(0.0)
                results_size.append([effective_size, hidden_size, 0.0])

        # Test 2: Varying Hidden Size (Fixed Total Size ~ 64M elements)
        print("\n--- 测试 2: Varying Hidden Size (Fixed Total Size ~ 64M) ---")
        hidden_sizes = [1024 * i for i in range(1, 17)] # 1024 to 16384
        target_total_elements = 64 * 1024 * 1024 # 64M
        bandwidths_hidden = []
        results_hidden = []
        
        for h in hidden_sizes:
            rows = target_total_elements // h
            shape = (rows, h)
            total_elements = rows * h
            
            print(f"测试 Hidden Size: {h} (Shape: {shape}, Total: {total_elements})")
            try:
                test_data = self.operator_test.generate_test_data(shape=shape)
                
                result = self.framework.run_core_operator_performance_test_v2(
                    operator_test=self.operator_test,
                    data=test_data,
                    device=device,
                    precision=PrecisionType.BF16, 
                    implementation="default",
                    num_warmup=10,
                    num_iterations=50
                )
                
                bw = result.bandwidth_gb_s
                if bw is None:
                    bw = 0.0
                
                bandwidths_hidden.append(bw)
                results_hidden.append([total_elements, h, bw])
                print(f"  带宽: {bw:.4f} GB/s")
                
            except Exception as e:
                print(f"  测试失败: {e}")
                bandwidths_hidden.append(0.0)
                results_hidden.append([total_elements, h, 0.0])

        # Save results to CSV
        csv_file_size = f"test_results/rmsnorm_bandwidth_size_{device.replace(':', '_')}_{timestamp}.csv"
        csv_file_hidden = f"test_results/rmsnorm_bandwidth_hidden_{device.replace(':', '_')}_{timestamp}.csv"
        
        try:
            with open(csv_file_size, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['TotalElements', 'HiddenSize', 'Bandwidth_GB_s'])
                writer.writerows(results_size)
            print(f"\n✅ 带宽测试数据(Size)已保存至 {csv_file_size}")
            
            with open(csv_file_hidden, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['TotalElements', 'HiddenSize', 'Bandwidth_GB_s'])
                writer.writerows(results_hidden)
            print(f"✅ 带宽测试数据(Hidden)已保存至 {csv_file_hidden}")
        except Exception as e:
            print(f"❌ 保存CSV失败: {e}")

        # 绘制曲线
        try:
            plt.figure(figsize=(20, 8))
            
            # Subplot 1: Bandwidth vs Total Size
            plt.subplot(1, 2, 1)
            plt.plot(sizes, bandwidths_size, 'bo-', linewidth=2, markersize=6)
            plt.xscale('log')
            plt.xlabel('Total Elements (log scale)')
            plt.ylabel('Bandwidth (GB/s)')
            plt.title(f'Bandwidth vs Total Size (Hidden=4096)')
            plt.grid(True, which="both", ls="-", alpha=0.5)
            for i, (size, bw) in enumerate(zip(sizes, bandwidths_size)):
                if i % 2 == 0 or i == len(sizes) - 1:
                    plt.annotate(f'{bw:.0f}', (size, bw), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

            # Subplot 2: Bandwidth vs Hidden Size
            plt.subplot(1, 2, 2)
            plt.plot(hidden_sizes, bandwidths_hidden, 'ro-', linewidth=2, markersize=6)
            plt.xlabel('Hidden Size (N)')
            plt.ylabel('Bandwidth (GB/s)')
            plt.title(f'Bandwidth vs Hidden Size (Total~64M)')
            plt.grid(True, which="both", ls="-", alpha=0.5)
            # Add user reference lines if needed, but let's stick to measured data first
            for i, (h, bw) in enumerate(zip(hidden_sizes, bandwidths_hidden)):
                 plt.annotate(f'{bw:.0f}', (h, bw), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            
            plt.tight_layout()
            output_file = f'test_results/rmsnorm_bandwidth_curve_{device.replace(":", "_")}_{timestamp}.png'
            plt.savefig(output_file)
            print(f"\n✅ 带宽曲线已保存至 {output_file}")
            plt.close()
            
        except Exception as e:
             print(f"❌ 绘图失败: {e}")

def main():
    """主函数 - 支持独立运行RMSNorm算子测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RMSNorm算子测试")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["accuracy", "performance", "comprehensive", "fulltest", "bandwidth"],
        default="comprehensive",
        help="测试模式 (默认: comprehensive)"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="test_results",
        help="测试结果保存目录 (默认: test_results)"
    )
    
    args = parser.parse_args()
    
    # 初始化测试框架和套件
    framework = OperatorTestFramework(result_dir=args.result_dir)
    suite = RMSNormTestSuite()
    suite.setup(framework)
    
    print(f"开始RMSNorm算子测试 (模式: {args.mode})")
    
    if args.mode == "accuracy":
        suite.run_accuracy_test()
    elif args.mode == "performance":
        suite.run_performance_test()
    elif args.mode == "comprehensive":
        # 综合测试：精度 + 性能
        suite.run_accuracy_test()
        suite.run_performance_test()
    elif args.mode == "fulltest":
        # 全面测试
        # Note: create_full_test_cases is not implemented yet, so we use create_test_cases
        print("注意: create_full_test_cases 未实现，使用 create_test_cases")
        cases = suite.create_test_cases()
        suite.run_accuracy_test(cases)
        suite.run_performance_test(cases)
    elif args.mode == "bandwidth":
        suite.run_bandwidth_test()

if __name__ == "__main__":
    main()
