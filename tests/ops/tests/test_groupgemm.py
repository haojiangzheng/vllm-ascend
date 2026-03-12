"""
GroupGemm算子测试套件
通用的 GroupGemm 测试，主要测试 num_experts=8 的场景
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
from tests.base_test_suite import BaseTestSuite
from operator_test_framework import PrecisionType
from groupgemm.groupgemm_int8 import GroupGemmOperatorTest
from groupgemm.groupgemm_bf16 import GroupGemmBF16OperatorTest


class GroupGemmTestSuite(BaseTestSuite):
    """GroupGemm算子测试套件 - 支持INT8和BF16精度"""
    
    def __init__(self, precision: str = "int8", num_experts: int = 8, hidden_dim: int = 7168, out_channel: int = 4096, use_nz_format: bool = False):
        """
        初始化GroupGemm测试套件
        
        Args:
            precision: 精度类型，"int8" 或 "bf16"
            num_experts: 专家数量
            hidden_dim: 隐藏维度
            out_channel: 输出通道数
            use_nz_format: 是否使用NZ格式（仅对INT8有效）
        """
        format_suffix = "_NZ" if use_nz_format else ""
        precision_name = f"GroupGemm_BF16{format_suffix}" if precision.lower() == "bf16" else f"GroupGemm{format_suffix}"
        super().__init__(precision_name)
        self.precision = precision.lower()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.out_channel = out_channel
        self.use_nz_format = use_nz_format
        self.operator_test = None
    
    def register_operator(self):
        """注册GroupGemm算子到测试框架"""
        if self.precision == "bf16":
            self.operator_test = GroupGemmBF16OperatorTest(
                num_experts=self.num_experts,
                hidden_dim=self.hidden_dim,
                out_channel=self.out_channel,
                use_nz_format=self.use_nz_format
            )
        else:
            self.operator_test = GroupGemmOperatorTest(
                num_experts=self.num_experts,
                hidden_dim=self.hidden_dim,
                out_channel=self.out_channel,
                use_nz_format=self.use_nz_format
            )
        self.framework.register_operator(self.operator_test)
    
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """创建标准测试案例 - GroupGemm主要使用Profile测试"""
        return self.create_profile_test_cases()
    
    def create_quick_test_cases(self) -> List[Dict[str, Any]]:
        """创建快速测试案例 - 使用较小的序列长度"""
        suffix = f"_{self.precision}" if self.precision == "bf16" else ""
        return [
            {
                'name': f'quick_profile2048{suffix}',
                'params': {
                    'seq_len': 2048,
                    'num_experts': self.num_experts
                }
            }
        ]
    
    def create_profile_test_cases(self) -> List[Dict[str, Any]]:
        """创建标准测试案例 - GroupGemm主要使用Profile测试"""
        return self.create_profile_test_cases()
    
    def create_profile_test_cases(self) -> List[Dict[str, Any]]:
        """创建专门的Profile测试案例"""
        suffix = f"_{self.precision}" if self.precision == "bf16" else ""
        return [
            {
                'name': f'profile2048{suffix}',
                'params': {
                    'seq_len': 2048,
                    'num_experts': self.num_experts
                }
            },
            {
                'name': f'profile4096{suffix}',
                'params': {
                    'seq_len': 4096,
                    'num_experts': self.num_experts
                }
            },
            {
                'name': f'profile8192{suffix}',
                'params': {
                    'seq_len': 8192,
                    'num_experts': self.num_experts
                }
            },
            {
                'name': f'profile16384{suffix}',
                'params': {
                    'seq_len': 16384,
                    'num_experts': self.num_experts
                }
            },
            {
                'name': f'profile32768{suffix}',
                'params': {
                    'seq_len': 32768,
                    'num_experts': self.num_experts
                }
            },
        ]
    
    def run_profile_test(self, test_cases: List[Dict[str, Any]] = None, num_iterations: int = 10):
        """运行 GroupGemm Profile 测试（使用基类增强版本）"""
        if test_cases is None:
            test_cases = self.create_profile_test_cases()
        
        # 显示 GroupGemm 特有信息
        precision_display = self.precision.upper()
        data_types = {
            "int8": "x=INT8, weight=INT8, bias=FP32, output=FP32",
            "bf16": "x=BF16, weight=BF16, bias=FP32, output=BF16"
        }
        
        print(f"GroupGemm {precision_display} 特有配置:")
        print(f"专家数量: {self.num_experts}")
        print(f"测试案例数: {len(test_cases)}")
        print(f"数据类型: {data_types.get(self.precision, 'Unknown')}")
        
        # 确定精度类型
        precision_type = PrecisionType.BF16 if self.precision == "bf16" else PrecisionType.INT8
        
        # 调用基类的增强版本，传入 GroupGemm 特有的参数
        return super().run_profile_test(
            test_cases=test_cases,
            num_iterations=num_iterations,
            precision_type=precision_type,
            filename_suffix=self.precision
        )
    

    
def main():
    """主函数 - 支持独立运行GroupGemm算子Profile测试"""
    import argparse
    from operator_test_framework import OperatorTestFramework
    
    parser = argparse.ArgumentParser(description='GroupGemm 算子 Profile 测试')
    parser.add_argument('--precision', choices=['int8', 'bf16'], default='int8', help='精度类型')
    parser.add_argument('--num-experts', type=int, default=8, help='专家数量')
    parser.add_argument('--hidden-dim', type=int, default=7168, help='隐藏维度')
    parser.add_argument('--out-channel', type=int, default=4096, help='输出通道')
    parser.add_argument('--device', default='npu:0', help='测试设备')
    parser.add_argument('--iterations', type=int, default=10, help='迭代次数')
    parser.add_argument('--use-nz-format', action='store_true', help='使用NZ格式（仅对INT8有效）')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['profile', 'performance'],
        default='profile',
        help='测试模式: profile(Profile测试) 或 performance(性能测试)'
    )
    parser.add_argument('--seq-lens', type=str, help='序列长度列表，用逗号分隔 (例如: 2048,4096,8192)')
    
    args = parser.parse_args()
    
    # 创建测试框架
    framework = OperatorTestFramework()
    
    # 根据精度类型创建测试套件
    test_suite = GroupGemmTestSuite(
        precision=args.precision,
        num_experts=args.num_experts,
        hidden_dim=args.hidden_dim,
        out_channel=args.out_channel,
        use_nz_format=args.use_nz_format
    )
    
    # 设置框架并注册算子
    test_suite.setup(framework)
    
    if args.precision == 'bf16':
        print(f"🔧 使用 BF16 精度测试 (x=BF16, weight=BF16, bias=FP32, output=BF16)")
    else:
        nz_info = " + NZ格式" if args.use_nz_format else ""
        print(f"🔧 使用 INT8 精度测试{nz_info} (x=INT8, weight=INT8, scale=BF16, per_token_scale=FP32, output=BF16)")
    
    try:
        # 创建自定义测试案例（如果指定了序列长度）
        test_cases = None
        if args.seq_lens:
            seq_lens = [int(x.strip()) for x in args.seq_lens.split(',')]
            suffix = f"_{args.precision}" if args.precision == "bf16" else ""
            test_cases = []
            for seq_len in seq_lens:
                test_cases.append({
                    'name': f'profile{seq_len}{suffix}',
                    'params': {
                        'seq_len': seq_len,
                        'num_experts': args.num_experts,
                        'hidden_dim': args.hidden_dim,
                        'out_channel': args.out_channel
                    }
                })
            print(f"📋 使用自定义序列长度: {seq_lens}")
        else:
            print(f"📋 使用默认Profile测试案例")
        
        # 根据模式运行相应测试
        if args.mode == "profile":
            print("🚀 运行 GroupGemm Profile 测试...")
            results = test_suite.run_profile_test(
                test_cases=test_cases,
                num_iterations=args.iterations
            )
            
            print(f"\n{'='*60}")
            print("✅ Profile 测试完成！")
            print(f"📁 结果已保存到 test_results 目录")
            print(f"📊 成功测试: {results['summary']['successful_tests']}/{results['summary']['total_tests']}")
            print(f"{'='*60}")
            
        elif args.mode == "performance":
            print("🚀 运行 GroupGemm 性能测试...")
            # 确定精度类型
            precision_type = PrecisionType.INT8 if args.precision == "int8" else PrecisionType.BF16
            results = test_suite.run_performance_test(
                test_cases=test_cases,
                precision_type=precision_type
            )
            
            print(f"\n{'='*60}")
            print("✅ 性能测试完成！")
            print(f"📁 结果已保存到 test_results 目录")
            print(f"📊 成功测试: {results['summary']['successful_tests']}/{results['summary']['total_tests']}")
            print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 强制清理NPU资源，防止Segmentation fault
        try:
            import torch_npu
            import gc
            import sys
            
            # 清空NPU缓存
            torch_npu.npu.empty_cache()
            
            # 强制垃圾回收
            gc.collect()
            
            # 同步NPU
            torch_npu.npu.synchronize()
            
            print("🧹 NPU资源清理完成")
            
        except Exception as cleanup_error:
            print(f"⚠️ 资源清理时出错: {cleanup_error}")
        
        # 显式退出，避免资源释放时的问题
        sys.exit(0)


if __name__ == "__main__":
    main()