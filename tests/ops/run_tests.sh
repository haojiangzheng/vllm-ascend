#!/bin/bash

# 算子测试框架运行脚本

echo "算子测试框架"
echo "============"

export TASK_QUEUE_ENABLE=2

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3"
    exit 1
fi

# 创建结果目录
mkdir -p test_results

echo ""
echo "可用的测试选项:"
echo "1. 只测试Add算子"
echo "2. 只测试PagedAttention算子"
echo "3. 只测试GroupGemm算子"
echo "4. 只测试Linear算子"
echo "5. 只测试RMSNorm算子"
echo "6. 列出所有已注册的算子"

read -p "请选择测试选项 (1-6): " choice

case $choice in
    1)
        echo "选择Add算子测试模式:"
        echo "a. 综合测试"
        echo "b. 只测试精度"
        echo "c. 只测试性能"
        echo "d. 全面随机测试(fulltest)"
        echo "e. 带宽测试"
        read -p "请选择模式 (a-e): " add_mode
        
        case $add_mode in
            a) python3 tests/test_add.py --mode comprehensive ;;
            b) python3 tests/test_add.py --mode accuracy ;;
            c) python3 tests/test_add.py --mode performance ;;
            d) python3 tests/test_add.py --mode fulltest ;;
            e) python3 tests/test_add.py --mode bandwidth ;;
            *) echo "无效选择"; exit 1 ;;
        esac
        ;;
    2)
        echo "选择PagedAttention算子测试模式:"
        echo "a. 综合测试"
        echo "b. 只测试精度"
        echo "c. 只测试性能"
        echo "d. Profile测试"
        echo "e. 全面随机测试(fulltest)"
        read -p "请选择模式 (a-e): " pa_mode
        
        case $pa_mode in
            a) python3 tests/test_paged_attention.py --mode comprehensive ;;
            b) python3 tests/test_paged_attention.py --mode accuracy ;;
            c) python3 tests/test_paged_attention.py --mode performance ;;
            d) python3 tests/test_paged_attention.py --mode profile ;;
            e) python3 tests/test_paged_attention.py --mode fulltest ;;
            *) echo "无效选择"; exit 1 ;;
        esac
        ;;
    3)
        echo "选择GroupGemm算子测试模式:"
        echo "a. INT8 Profile测试"
        echo "b. INT8 性能测试"
        echo "c. BF16 Profile测试"
        echo "d. BF16 性能测试"
        echo "e. INT8 + NZ格式 Profile测试"
        echo "f. INT8 + NZ格式 性能测试"
        read -p "请选择模式 (a-f): " gg_mode
        
        case $gg_mode in
            a) python3 tests/test_groupgemm.py --precision int8 --mode profile ;;
            b) python3 tests/test_groupgemm.py --precision int8 --mode performance ;;
            c) python3 tests/test_groupgemm.py --precision bf16 --mode profile ;;
            d) python3 tests/test_groupgemm.py --precision bf16 --mode performance ;;
            e) python3 tests/test_groupgemm.py --precision int8 --use-nz-format --mode profile ;;
            f) python3 tests/test_groupgemm.py --precision int8 --use-nz-format --mode performance ;;
            *) echo "无效选择"; exit 1 ;;
        esac
        ;;
    4)
        echo "选择Linear算子测试模式:"
        echo "a. Profile测试 - FP16"
        echo "b. Profile测试 - BF16"
        echo "c. 性能测试 V2 - FP16"
        echo "d. 性能测试 V2 - BF16"
        echo "e. TFLOPS测试 - FP16"
        echo "f. TFLOPS测试 - BF16"

        read -p "请选择模式 (a-f): " linear_mode
        
        case $linear_mode in
            a) python3 tests/test_linear.py --precision fp16 --mode profile ;;
            b) python3 tests/test_linear.py --precision bf16 --mode profile ;;
            c) python3 tests/test_linear.py --precision fp16 --mode performance ;;
            d) python3 tests/test_linear.py --precision bf16 --mode performance ;;
            e) python3 tests/test_linear.py --precision fp16 --mode tflops ;;
            f) python3 tests/test_linear.py --precision bf16 --mode tflops ;;

            *) echo "无效选择"; exit 1 ;;
        esac
        ;;
    5)
        echo "选择RMSNorm算子测试模式:"
        echo "a. 综合测试"
        echo "b. 只测试精度"
         echo "c. 只测试性能"
         echo "d. 全面随机测试(fulltest)"
         echo "e. 带宽测试"
         read -p "请选择模式 (a-e): " rms_mode
         
         case $rms_mode in
             a) python3 tests/test_rmsnorm.py --mode comprehensive ;;
             b) python3 tests/test_rmsnorm.py --mode accuracy ;;
             c) python3 tests/test_rmsnorm.py --mode performance ;;
             d) python3 tests/test_rmsnorm.py --mode fulltest ;;
             e) python3 tests/test_rmsnorm.py --mode bandwidth ;;
             *) echo "无效选择"; exit 1 ;;
         esac
         ;;
    6)
        echo "列出所有已注册的算子..."
        python3 test_main.py --list
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "测试完成！结果保存在 test_results/ 目录中"