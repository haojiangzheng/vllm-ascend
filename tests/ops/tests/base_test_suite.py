"""
测试套件基类
定义统一的测试接口
"""

import time
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
from operator_test_framework import (
    OperatorTestFramework, 
    PrecisionType,
    ProfilerBackend,
    ProfilerConfig, 
    ProfilerFactory
)

class BaseTestSuite(ABC):
    """测试套件基类"""
    
    def __init__(self, operator_name: str):
        self.operator_name = operator_name
        self.framework = None
    
    def setup(self, framework: OperatorTestFramework):
        """设置测试框架"""
        self.framework = framework
        self.register_operator()
    
    @abstractmethod
    def register_operator(self):
        """注册算子到测试框架"""
        pass
    
    @abstractmethod
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """创建测试案例"""
        pass
    
    @abstractmethod
    def create_quick_test_cases(self) -> List[Dict[str, Any]]:
        """创建快速测试案例"""
        pass
    
    def run_accuracy_test(self, test_cases: List[Dict[str, Any]] = None):
        """运行精度测试"""
        if test_cases is None:
            test_cases = self.create_test_cases()
        
        print(f"\n{'='*80}")
        print(f"{self.operator_name}算子精度测试")
        print(f"{'='*80}")
        
        all_accuracy_results = []
        
        for i, test_case in enumerate(test_cases):
            test_name = test_case.get('name', f'test_case_{i}')
            test_params = test_case.get('params', {})
            
            try:
                # 生成测试数据
                operator_test = self.framework.operators[self.operator_name]
                test_data = operator_test.generate_test_data(**test_params)
                
                # 运行精度测试
                accuracy_result = self.framework.run_accuracy_test(self.operator_name, test_data, test_name)
                all_accuracy_results.append(accuracy_result)
                
            except Exception as e:
                print(f"❌ 测试案例 '{test_name}' 失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # 保存精度测试结果
        accuracy_results = {
            'operator_name': self.operator_name,
            'test_type': 'accuracy_only',
            'accuracy_results': all_accuracy_results,
            'summary': self.framework._generate_summary(all_accuracy_results, [])
        }
        
        self.framework.save_results(accuracy_results, f"{self.operator_name.lower()}_accuracy_test")
        return accuracy_results
    
    def run_performance_test(self, test_cases: List[Dict[str, Any]] = None, include_core_analysis: bool = False, precision_type: PrecisionType = None):
        """运行性能测试
        
        Args:
            test_cases: 测试案例列表
            include_core_analysis: 是否包含核心算子开销分析（已弃用，保留兼容性）
            precision_type: 指定精度类型
        """
        if test_cases is None:
            test_cases = self.create_test_cases()
        
        # 统一使用 V2 性能测试方法
        return self.run_performance_test_suite_v2(test_cases=test_cases, precision_type=precision_type)
        

    
    def run_profile_test(self, test_cases: List[Dict[str, Any]] = None, num_iterations: int = 10, 
                        precision_type: PrecisionType = None, filename_suffix: str = None):
        """运行 Profile 测试
        
        Args:
            test_cases: 测试案例列表
            num_iterations: 迭代次数，默认10次
            precision_type: 精度类型，如果不指定则从测试数据中获取
            filename_suffix: 文件名后缀，如果不指定则使用默认格式
        """
        if test_cases is None:
            # 优先使用专门的 profile 测试案例
            if hasattr(self, 'create_profile_test_cases'):
                test_cases = self.create_profile_test_cases()
            else:
                test_cases = self.create_test_cases()
        
        print(f"\n{'='*80}")
        print(f"{self.operator_name}算子 Profile 测试")
        if num_iterations != 10:
            print(f"迭代次数: {num_iterations}")
        if precision_type:
            print(f"精度类型: {precision_type.name}")
        print(f"{'='*80}")
        
        all_profile_results = []
        
        for i, test_case in enumerate(test_cases):
            test_name = test_case.get('name', f'test_case_{i}')
            test_params = test_case.get('params', {})
            
            try:
                # 生成测试数据
                operator_test = self.framework.operators[self.operator_name]
                test_data = operator_test.generate_test_data(**test_params)
                
                # 确定使用的精度类型
                used_precision = precision_type or test_data.get('precision', PrecisionType.BF16)
                
                # 运行统一 profile 测试
                profile_result = self.framework.run_unified_profile_test(
                    operator_name=self.operator_name,
                    test_data=test_data,
                    device="npu:0",  # 默认使用 NPU
                    precision=used_precision,
                    num_warmup=5,
                    num_iterations=num_iterations,
                    enable_profile=True,
                    test_name=test_name
                )
                all_profile_results.append(profile_result)
                
                print(f"✅ Profile 测试案例 '{test_name}' 完成")
                
            except Exception as e:
                print(f"❌ Profile 测试案例 '{test_name}' 失败: {str(e)}")
                continue
        
        # 保存 Profile 测试结果
        profile_results = {
            'operator_name': self.operator_name,
            'test_type': 'profile_only',
            'profile_results': all_profile_results,
            'summary': {
                'total_tests': len(test_cases),
                'successful_tests': len(all_profile_results),
                'failed_tests': len(test_cases) - len(all_profile_results)
            }
        }
        
        # 使用自定义文件名后缀或默认格式
        if filename_suffix:
            filename = f"{self.operator_name.lower()}_{filename_suffix}_profile_test"
        else:
            filename = f"{self.operator_name.lower()}_profile_test"
            
        self.framework.save_results(profile_results, filename)
        return profile_results
    
    def run_comprehensive_test(self, test_cases: List[Dict[str, Any]] = None):
        """运行综合测试（精度+性能）"""
        if test_cases is None:
            test_cases = self.create_test_cases()
        
        print(f"\n{'='*80}")
        print(f"{self.operator_name}算子综合测试")
        print(f"{'='*80}")
        
        all_accuracy_results = []
        all_performance_results = []
        
        for i, test_case in enumerate(test_cases):
            test_name = test_case.get('name', f'test_case_{i}')
            test_params = test_case.get('params', {})
            
            try:
                # 生成测试数据
                operator_test = self.framework.operators[self.operator_name]
                test_data = operator_test.generate_test_data(**test_params)
                
                # 运行精度测试
                accuracy_result = self.framework.run_accuracy_test(self.operator_name, test_data, test_name)
                all_accuracy_results.append(accuracy_result)
                
                # 运行性能测试 - 使用V1版本的测试套件方法
                performance_result = self._run_single_case_performance_test_v1(test_data, test_name)
                all_performance_results.append(performance_result)
                
            except Exception as e:
                print(f"❌ 测试案例 '{test_name}' 失败: {str(e)}")
                continue
        
        # 保存结果
        comprehensive_results = {
            'operator_name': self.operator_name,
            'accuracy_results': all_accuracy_results,
            'performance_results': all_performance_results,
            'summary': self.framework._generate_summary(all_accuracy_results, all_performance_results)
        }
        
        self.framework.save_results(comprehensive_results, f"{self.operator_name}_comprehensive_test")
        self.framework.print_summary_report(comprehensive_results)
        
        return comprehensive_results
    
    def _run_single_case_performance_test_v1(self, test_data: Dict[str, Any], test_name: str) -> Dict[str, Any]:
        """为单个测试案例运行V1性能测试"""
        operator_test = self.framework.operators[self.operator_name]
        
        print(f"\n{'='*60}")
        print(f"性能测试: {self.operator_name} - {test_name}")
        print(f"{'='*60}")
        
        results = {
            'operator_name': self.operator_name,
            'test_name': test_name,
            'test_data_info': test_data.get('metadata', {}),
            'performance_results': {}
        }
        
        # 对所有支持的精度进行性能测试
        for precision in operator_test.supported_precisions:
            precision_name = precision.name
            print(f"\n=== 性能测试精度: {precision_name} ===")
            
            precision_results = {}
            
            for device_type in operator_test.supported_devices:
                device = device_type.value
                if device == "npu":
                    device = "npu:0"
                elif device == "cuda":
                    device = "cuda:0"
                
                device_results = {}
                implementations = operator_test.get_available_implementations(device)
                
                for impl in implementations:
                    try:
                        # 完整函数性能测试
                        perf_metrics = self.framework.run_performance_test(
                            operator_test, test_data, device, precision, impl
                        )
                        
                        impl_results = {
                            'full_function': {
                                'metrics': perf_metrics,
                                'success': True
                            }
                        }
                        
                        # 核心算子性能测试 (使用 V2)
                        try:
                            core_perf_metrics = self.framework.run_core_operator_performance_test_v2(
                                operator_test, test_data, device, precision, impl
                            )
                            impl_results['core_operator'] = {
                                'metrics': core_perf_metrics,
                                'success': True
                            }
                        except Exception as core_e:
                            print(f"    ⚠️  核心算子性能测试失败: {str(core_e)}")
                            impl_results['core_operator'] = {
                                'metrics': None,
                                'success': False,
                                'error': str(core_e)
                            }
                        
                        device_results[impl] = impl_results
                        
                    except Exception as e:
                        print(f"    ❌ 性能测试失败: {str(e)}")
                        device_results[impl] = {
                            'full_function': {
                                'metrics': None,
                                'success': False,
                                'error': str(e)
                            }
                        }
                
                precision_results[device] = device_results
            
            results['performance_results'][precision_name] = precision_results
        
        return results
    
    def _run_performance_test_suite_common(
        self,
        test_cases: List[Dict[str, Any]],
        version: str,
        precision_type: PrecisionType = None,
        test_method_name: str = None,
        result_key: str = 'core_performance',
        description: str = None,
        num_warmup: int = 10,
        num_iterations: int = 20,
        extra_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """通用的性能测试套件执行逻辑
        
        Args:
            test_cases: 测试案例列表
            version: 版本标识 (v2)
            precision_type: 精度类型
            test_method_name: 要调用的测试方法名
            result_key: 结果字典中的键名
            description: 版本描述信息
            num_warmup: 预热次数
            num_iterations: 测试迭代次数
            extra_params: 额外参数
        
        Returns:
            Dict: 性能测试结果
        """
        # 构建测试方法参数
        test_method_kwargs = {
            'num_warmup': num_warmup,
            'num_iterations': num_iterations
        }
        
        # 添加额外参数
        if extra_params:
            test_method_kwargs.update(extra_params)
        
        # 打印测试开始信息
        print(f"\n{'='*80}")
        print(f"{self.operator_name}算子核心性能测试 {version.upper()}")
        if description:
            print(description)
        if precision_type:
            print(f"精度类型: {precision_type.name}")
        print(f"{'='*80}")
        
        core_performance_results = []
        
        for i, test_case in enumerate(test_cases):
            test_name = test_case.get('name', f'test_case_{i}')
            test_params = test_case.get('params', {})
            
            print(f"\n🎯 测试用例: {test_name}")
            print("-" * 60)
            
            try:
                # 生成测试数据
                operator_test = self.framework.operators[self.operator_name]
                test_data = operator_test.generate_test_data(**test_params)
                
                # 确定使用的精度类型
                used_precision = (precision_type or 
                                test_case.get('precision') or 
                                test_data.get('precision', PrecisionType.BF16))
                
                # 获取可用的设备和实现
                devices = ["npu:0"]  # 主要测试NPU
                implementations = operator_test.get_available_implementations("npu:0")
                
                for device in devices:
                    for impl in implementations:
                        print(f"\n📊 测试配置: {device} - {used_precision.name} - {impl}")
                        
                        try:
                            # 调用指定的测试方法
                            test_method = getattr(self.framework, test_method_name)
                            core_metrics = test_method(
                                operator_test=operator_test,
                                data=test_data,
                                device=device,
                                precision=used_precision,
                                implementation=impl,
                                **test_method_kwargs
                            )
                            
                            # 构建结果记录
                            core_result = {
                                'test_name': test_name,
                                'device': device,
                                'implementation': impl,
                                'precision': used_precision.name,
                                'test_params': test_params
                            }
                            
                            # V2 使用标准的 metrics 对象
                            core_result['core_performance_v2'] = {
                                'avg_time_ms': core_metrics.avg_time_ms,
                                'iterations': core_metrics.iterations,
                                'throughput_ops_per_sec': core_metrics.throughput_ops_per_sec,
                                    'tops': core_metrics.tops,
                                    'bandwidth_gb_s': core_metrics.bandwidth_gb_s
                                }
                            core_result['warmup_iterations'] = test_method_kwargs.get('num_warmup', 10)
                            
                            core_performance_results.append(core_result)
                            
                            # 打印测试结果
                            print(f"✅ 核心性能测试 {version.upper()} '{test_name}' ({device}-{impl}) 完成")
                            print(f"  🚀 核心算子: {core_metrics.avg_time_ms:.3f}ms")
                            
                            # 打印 V2 性能指标
                            if core_metrics.tops is not None:
                                print(f"  📊 计算性能: {core_metrics.tops:.2f} TOPS")
                            if core_metrics.bandwidth_gb_s is not None:
                                print(f"  🌊 内存带宽: {core_metrics.bandwidth_gb_s:.2f} GB/s")
                            if core_metrics.throughput_ops_per_sec is not None:
                                print(f"  📈 吞吐量: {core_metrics.throughput_ops_per_sec:.2f} ops/sec")
                                
                        except Exception as e:
                            error_msg = f"核心算子{version.upper()}测试失败 ({device}-{impl}): {str(e)}"
                            print(f"    ❌ {error_msg}")
                            continue
                
                # 每个测试案例完成后清理NPU内存
                try:
                    import torch_npu
                    torch_npu.npu.empty_cache()
                except Exception:
                    pass  # 忽略清理错误
                
            except Exception as e:
                print(f"❌ 测试案例 '{test_name}' 失败: {str(e)}")
                # 即使测试失败也要清理内存
                try:
                    import torch_npu
                    torch_npu.npu.empty_cache()
                except Exception:
                    pass
                continue
        
        # 构建和返回结果
        return self._build_performance_test_results(
            test_cases, core_performance_results, version, result_key, test_method_kwargs
        )
    
    def _build_performance_test_results(
        self,
        test_cases: List[Dict[str, Any]],
        core_performance_results: List[Dict[str, Any]],
        version: str,
        result_key: str,
        test_method_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """构建性能测试结果 - 统一的结果结构"""
        
        # 基础结果结构
        results = {
            f'core_performance_results_{version}': core_performance_results,
            'summary': {
                'total_tests': len(test_cases),
                'successful_tests': len(core_performance_results),
                'failed_tests': len(test_cases) - len(core_performance_results),
                'operator_name': self.operator_name,
                'version': version,
                'warmup_iterations': test_method_kwargs.get('num_warmup', 10),
                'test_iterations': test_method_kwargs.get('num_iterations', 100)
            }
        }
        
        # 打印汇总信息
        if core_performance_results:
            print(f"\n📊 {version.upper()} 性能测试汇总:")
            print(f"  ✅ 成功测试: {results['summary']['successful_tests']}/{results['summary']['total_tests']}")
            
            # 计算统计信息 - V2 版本从 core_performance_v2 中获取时间
            valid_times = []
            for r in core_performance_results:
                if 'core_performance_v2' in r:
                    valid_times.append(r['core_performance_v2']['avg_time_ms'])
            
            if valid_times:
                print(f"  ⚡ 平均核心时间: {sum(valid_times)/len(valid_times):.3f}ms")
                print(f"  🏃 最快核心时间: {min(valid_times):.3f}ms")
                print(f"  🐌 最慢核心时间: {max(valid_times):.3f}ms")
        
        # 保存结果
        filename = f"{self.operator_name.lower()}_performance_{version}"
        self.framework.save_results(results, filename)
        
        # 清理NPU内存
        try:
            import torch_npu
            import gc
            torch_npu.npu.empty_cache()
            gc.collect()
            print(f"✓ 测试结果已保存到: test_results/{filename}.json")
        except Exception as e:
            print(f"警告: NPU内存清理失败: {e}")
        
        return results

    def run_performance_test_suite_v2(self, test_cases: List[Dict[str, Any]], precision_type: PrecisionType = None, num_warmup: int = 10, num_iterations: int = 20):
        """运行性能测试套件 V2版本（预先准备所有数据，避免数据准备开销）
        
        Args:
            test_cases: 测试案例列表
            precision_type: 精度类型
            num_warmup: 预热次数
            num_iterations: 测试迭代次数
        
        Returns:
            Dict: 核心算子性能测试结果
        """
        return self._run_performance_test_suite_common(
            version="v2",
            test_method_name="run_core_operator_performance_test_v2",
            test_cases=test_cases,
            precision_type=precision_type,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
            extra_params={},
            result_key='core_performance_results_v2',
            description="预先准备数据，零开销性能测试"
        )



    def run_quick_test(self):
        """运行快速测试"""
        quick_test_cases = self.create_quick_test_cases()
        return self.run_comprehensive_test(quick_test_cases)