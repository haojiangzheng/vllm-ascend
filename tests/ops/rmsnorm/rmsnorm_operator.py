import torch
try:
    import torch_npu
except ImportError:
    torch_npu = None
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
import sys
import os

# Add parent directory to path to import framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from operator_test_framework import BaseOperatorTest, PrecisionType, DeviceType


class RMSNormOperatorTest(BaseOperatorTest):
    """RMSNorm算子测试实现"""
    
    def __init__(self):
        super().__init__("RMSNorm")
        self.supported_precisions = [PrecisionType.FP16, PrecisionType.BF16, PrecisionType.FP32]
        self.supported_devices = []
        if torch_npu is not None:
            self.supported_devices.append(DeviceType.NPU)
    
    def generate_test_data(
        self,
        shape: tuple = (32, 2048),  # typical batch size and hidden size
        eps: float = 1e-6,
        value_range: tuple = (-1.0, 1.0),
        **kwargs
    ) -> Dict[str, Any]:
        """生成RMSNorm测试数据"""
        
        # input tensor
        x = torch.rand(shape) * (value_range[1] - value_range[0]) + value_range[0]
        
        # gamma (scale) parameter, usually initialized to 1s or random
        # The last dimension is the normalized dimension
        normalized_shape = shape[-1]
        gamma = torch.rand(normalized_shape) * (value_range[1] - value_range[0]) + value_range[0]
        
        return {
            'x': x,
            'gamma': gamma,
            'eps': eps,
            'metadata': {
                'shape': shape,
                'eps': eps,
                'operator_type': 'RMSNorm',
                'total_elements': x.numel(),
                # RMSNorm flops: per element: square, sum (reduction), sqrt, div, mul
                # approx 4 ops per element depending on implementation details
                'flops': x.numel() * 4 
            }
        }
    
    def run_cpu_reference(self, data: Dict[str, Any]) -> torch.Tensor:
        """运行CPU参考实现"""
        x = data['x']
        gamma = data['gamma']
        eps = data['eps']
        
        # Manual RMSNorm implementation
        # x: [..., d]
        # rms = sqrt(mean(x**2) + eps)
        # y = x / rms * gamma
        
        # Cast to float32 for higher precision reference calculation
        x_float = x.float()
        gamma_float = gamma.float()
        
        mean_square = torch.mean(x_float ** 2, dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + eps)
        result = x_float * torch.rsqrt(mean_square + eps) * gamma_float
        
        return result
    
    def run_device_implementation(
        self,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default"
    ) -> torch.Tensor:
        """运行设备实现"""
        
        # Move data to device and cast to precision
        x = data['x'].to(device=device, dtype=precision.value)
        gamma = data['gamma'].to(device=device, dtype=precision.value)
        eps = data['eps']
        
        if implementation == "default":
            if "npu" in device and torch_npu is not None and hasattr(torch_npu, 'npu_rms_norm'):
                 # Signature: npu_rms_norm(Tensor self, Tensor gamma, float epsilon=1e-06) -> (Tensor, Tensor)
                 # Returns (y, rstd)
                 y, _ = torch_npu.npu_rms_norm(x, gamma, epsilon=eps)
                 return y.cpu().float()
            else:
                 # Fallback to manual if NPU op not available or on CPU
                 pass

        # Manual implementation (works for CPU and NPU if custom op not available)
        # implementation == "manual" or fallback from default
        mean_square = torch.mean(x ** 2, dim=-1, keepdim=True)
        # Use rsqrt for potentially better performance/stability
        result = x * torch.rsqrt(mean_square + eps) * gamma
            
        return result.cpu().float()

    def _prepare_data_for_core_operator(self, data: Dict[str, Any], device: str, precision: PrecisionType, implementation: str = "default") -> Dict[str, Any]:
        """为核心算子准备数据（排除预处理开销）"""
        x = data['x'].to(device=device, dtype=precision.value)
        gamma = data['gamma'].to(device=device, dtype=precision.value)
        
        return {
            'x': x,
            'gamma': gamma,
            'eps': data['eps'],
            'implementation': implementation,
            'device': device
        }

    def _execute_core_operator(self, prepared_data: Dict[str, Any], implementation: str = "default") -> torch.Tensor:
        """执行核心算子（只测量核心计算，不包括数据移动）"""
        x = prepared_data['x']
        gamma = prepared_data['gamma']
        eps = prepared_data['eps']
        device = prepared_data.get('device', '')
        impl = prepared_data.get('implementation', implementation)
        
        if impl == "default":
            if "npu" in device and torch_npu is not None and hasattr(torch_npu, 'npu_rms_norm'):
                y, _ = torch_npu.npu_rms_norm(x, gamma, epsilon=eps)
                return y
        
        # Manual implementation
        mean_square = torch.mean(x ** 2, dim=-1, keepdim=True)
        result = x * torch.rsqrt(mean_square + eps) * gamma
        return result

    def get_available_implementations(self, device: str) -> List[str]:
        """获取可用的实现列表"""
        if "npu" in device and torch_npu is not None and hasattr(torch_npu, 'npu_rms_norm'):
            return ["default"]
        return ["manual"]
    
    def calculate_throughput(self, data: Dict[str, Any], time_ms: float) -> Optional[float]:
        """计算吞吐量（GFLOPS）"""
        # Approx ops: square, mean(sum+div), add eps, sqrt, div, mul
        # Let's say 4 ops per element for simplicity
        flops = data['metadata']['flops']
        return (flops / (time_ms / 1000)) / 1e9  # GFLOPS
        
    def calculate_bandwidth(self, data: Dict[str, Any], avg_time_ms: float, precision: PrecisionType = None) -> Optional[float]:
        """计算内存带宽（GB/s）"""
        # Read x (once), Read gamma (broadcast, negligible for large batch), Write y (once)
        # Total bytes = 2 * num_elements * sizeof(dtype)
        
        num_elements = data['metadata']['total_elements']
        dtype_size = self.get_precision_config(precision)['dtype_size']
        
        total_bytes = 2 * num_elements * dtype_size
        
        return (total_bytes / (avg_time_ms / 1000)) / 1e9 # GB/s

    def get_precision_config(self, precision: PrecisionType = None) -> Dict[str, int]:
        """获取精度配置"""
        if precision == PrecisionType.FP16:
            return {'dtype_size': 2}
        elif precision == PrecisionType.BF16:
            return {'dtype_size': 2}
        elif precision == PrecisionType.FP32:
            return {'dtype_size': 4}
        else:
            # 默认为2字节 (FP16/BF16)
            return {'dtype_size': 2}
