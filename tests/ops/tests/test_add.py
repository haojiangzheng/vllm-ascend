"""
Add算子测试套件
"""

import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
from tests.base_test_suite import BaseTestSuite
from add.add_operator import AddOperatorTest


class AddTestSuite(BaseTestSuite):
    """Add算子测试套件"""
    
    def __init__(self):
        super().__init__("Add")
        self.operator_test = AddOperatorTest()
    
    def register_operator(self):
        """注册Add算子到测试框架"""
        self.framework.register_operator(self.operator_test)
    
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """创建Add算子测试案例"""
        return [
            {
                'name': 'large_tensor_wide_range',
                'params': {
                    'shape': (1024, 1024),
                    'value_range': (-1000.0, 1000.0)
                }
            },
            {
                'name': 'large_rectangular',
                'params': {
                    'shape': (512, 2048),
                    'value_range': (-1.0, 1.0)
                }
            },
            {
                'name': 'xlarge_tensor',
                'params': {
                    'shape': (2048, 2048),
                    'value_range': (-1.0, 1.0)
                }
            }
        ]
    
    def create_quick_test_cases(self) -> List[Dict[str, Any]]:
        """创建快速测试案例"""
        return [
            {
                'name': 'quick_medium',
                'params': {
                    'shape': (512, 512),
                    'value_range': (-10.0, 10.0)
                }
            },
        ]
    
    def create_full_test_cases(self, num_cases: int = 50) -> List[Dict[str, Any]]:
        """创建全面测试案例（随机shape与数值范围）"""
        test_cases = []
        preset_shapes = [
            (256, 256), (512, 512), (1024, 512),
            (1024, 1024), (2048, 1024), (512, 2048), (2048, 2048)
        ]
        value_ranges = [(-1.0, 1.0), (-10.0, 10.0), (-1000.0, 1000.0)]
        
        for i in range(num_cases):
            if random.random() < 0.5:
                shape = random.choice(preset_shapes)
            else:
                # 随机二维shape，避免过大导致测试时间过长
                shape = (random.randint(128, 4096), random.randint(128, 4096))
            value_range = random.choice(value_ranges)
            test_cases.append({
                'name': f'full_test_case_{i+1}',
                'params': {
                    'shape': shape,
                    'value_range': value_range
                }
            })
        return test_cases

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

        # 2^12 到 2^27
        sizes = [2**i for i in range(12, 28)]
        bandwidths = []
        
        print(f"\n{'='*80}")
        print("Add算子带宽测试 (2^12 - 2^27)")
        print(f"{'='*80}")
        
        # 确定设备
        device = "npu:0" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"
        if device == "cpu" and torch.cuda.is_available():
            device = "cuda:0"
        elif device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            
        print(f"使用设备: {device}")

        from operator_test_framework import PrecisionType
        
        # Ensure test_results directory exists
        os.makedirs("test_results", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_file = f"test_results/add_bandwidth_{device.replace(':', '_')}_{timestamp}.csv"
        
        results_data = []

        for size in sizes:
            print(f"测试大小: {size}")
            try:
                # 生成测试数据 - 使用1D tensor
                shape = (size,)
                test_data = self.operator_test.generate_test_data(shape=shape)
                
                # 运行性能测试
                result = self.framework.run_core_operator_performance_test_v2(
                    operator_test=self.operator_test,
                    data=test_data,
                    device=device,
                    precision=PrecisionType.BF16, # 默认使用BF16
                    implementation="default",
                    num_warmup=5,
                    num_iterations=20
                )
                
                bw = result.bandwidth_gb_s
                if bw is None:
                    bw = 0.0
                
                bandwidths.append(bw)
                results_data.append([size, bw])
                print(f"  带宽: {bw:.4f} GB/s")
                
            except Exception as e:
                print(f"  测试失败: {e}")
                bandwidths.append(0.0)
                results_data.append([size, 0.0])
                import traceback
                traceback.print_exc()

        # Save results to CSV
        try:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Size', 'Bandwidth_GB_s'])
                writer.writerows(results_data)
            print(f"\n✅ 带宽测试数据已保存至 {csv_file}")
        except Exception as e:
            print(f"❌ 保存CSV失败: {e}")

        # 绘制曲线
        try:
            plt.figure(figsize=(12, 7))  # 稍微加大一点画布
            plt.plot(sizes, bandwidths, 'bo-', linewidth=2, markersize=6)
            plt.xscale('log')
            plt.xlabel('Tensor Size (elements)')
            plt.ylabel('Bandwidth (GB/s)')
            plt.title(f'Add Operator Bandwidth vs Tensor Size ({device})')
            plt.grid(True, which="both", ls="-", alpha=0.5)
            
            # 设置横坐标刻度显示为 10^x 格式，虽然点是 2^x 分布的
            # matplotlib log scale默认会自动处理好刻度，这里不需要额外强制设置，
            # 除非为了完全符合"1e4, 1e5"的视觉要求，可以不做特殊处理，默认的log显示通常就是10^x
            
            # 添加数值标注 (隔几个点标注一下，避免太密集)
            for i, (size, bw) in enumerate(zip(sizes, bandwidths)):
                # 只标注部分点，或者全部标注但字体小一点
                if i % 2 == 0 or i == len(sizes) - 1:
                    plt.annotate(f'{bw:.1f}', (size, bw), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

            output_file = f'test_results/add_bandwidth_curve_{device.replace(":", "_")}_{timestamp}.png'
            plt.savefig(output_file)
            print(f"✅ 带宽曲线已保存至 {output_file}")
            plt.close()
        except Exception as e:
             print(f"❌ 绘图失败: {e}")

def main():
    """主函数 - 支持独立运行Add算子测试"""
    import argparse
    import sys
    import os
    
    # 添加父目录到Python路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from operator_test_framework import OperatorTestFramework
    
    parser = argparse.ArgumentParser(description="Add算子测试")
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
    parser.add_argument(
        "--num-cases",
        type=int,
        default=50,
        help="fulltest模式下生成的随机测试案例数量 (默认: 50)"
    )
    
    args = parser.parse_args()
    
    # 设置测试框架
    framework = OperatorTestFramework(result_dir=args.result_dir)
    
    # 创建并设置Add测试套件
    add_suite = AddTestSuite()
    add_suite.setup(framework)
    
    # 根据模式运行测试
    try:
        if args.mode == "accuracy":
            print("🧪 运行 Add 算子精度测试...")
            results = add_suite.run_accuracy_test()
        elif args.mode == "performance":
            print("🎯 运行 Add 算子性能测试（专注算子本身，忽略Python下发开销）...")
            from operator_test_framework import PrecisionType
            results = add_suite.run_performance_test(precision_type=PrecisionType.BF16)
        elif args.mode == "comprehensive":
            print("🔍 运行 Add 算子综合测试...")
            results = add_suite.run_comprehensive_test()
        elif args.mode == "fulltest":
            print("🎯 运行 Add 算子全面随机测试...")
            full_test_cases = add_suite.create_full_test_cases(num_cases=args.num_cases)
            results = add_suite.run_comprehensive_test(full_test_cases)
        elif args.mode == "bandwidth":
            print("📈 运行 Add 算子带宽测试...")
            results = add_suite.run_bandwidth_test()
        else:
            print(f"❌ 不支持的测试模式: {args.mode}")
            print("支持的模式: accuracy, performance, comprehensive, fulltest, bandwidth")
            return 1
        
        print(f"\n✅ Add算子测试完成，结果已保存到 {args.result_dir}")
        return 0
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()
