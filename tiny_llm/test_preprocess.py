import os
import sys
import torch
import numpy as np

# 确保能导入 preprocess 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import PretokDataset, Task


def test_pretok_dataset():
    """测试 PretokDataset 类的功能"""
    print("=== 测试 PretokDataset 类 ===")
    
    # 测试参数
    max_seq_len = 256
    vocab_size = 4096
    vocab_source = 'custom'
    
    try:
        # 初始化训练数据集
        print("\n1. 初始化训练数据集...")
        train_dataset = PretokDataset(
            split='train', 
            max_seq_len=max_seq_len, 
            vocab_size=vocab_size, 
            vocab_source=vocab_source
        )
        print("✓ 训练数据集初始化成功!")
        
        # 初始化测试数据集
        print("\n2. 初始化测试数据集...")
        test_dataset = PretokDataset(
            split='test', 
            max_seq_len=max_seq_len, 
            vocab_size=vocab_size, 
            vocab_source=vocab_source
        )
        print("✓ 测试数据集初始化成功!")
        
        # 测试迭代功能 - 只迭代几个批次用于测试
        print("\n3. 测试数据集迭代功能...")
        train_iterator = iter(train_dataset)
        
        # 获取第一个批次
        first_x, first_y = next(train_iterator)
        print(f"✓ 成功获取第一个批次!")
        print(f"输入形状: {first_x.shape}")
        print(f"标签形状: {first_y.shape}")
        print(f"数据类型: {first_x.dtype}")
        
        # 验证批次大小是否正确
        assert first_x.shape[0] == max_seq_len, f"期望序列长度为 {max_seq_len}，实际为 {first_x.shape[0]}"
        assert first_y.shape[0] == max_seq_len, f"期望标签长度为 {max_seq_len}，实际为 {first_y.shape[0]}"
        print("✓ 序列长度验证通过!")
        
        # 验证 x 和 y 的关系 (y 应该是 x 右移一位)
        assert torch.allclose(first_x[1:], first_y[:-1]), "x 和 y 不是移位关系"
        print("✓ x 和 y 移位关系验证通过!")
        
        # 获取第二个批次，确保迭代器正常工作
        second_x, second_y = next(train_iterator)
        print("✓ 成功获取第二个批次!")
        
        print("\nPretokDataset 类测试全部通过!")
        return True
        
    except Exception as e:
        print(f"PretokDataset 类测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task():
    """测试 Task 类的功能"""
    print("\n=== 测试 Task 类 ===")
    
    # 测试参数
    batch_size = 4
    max_seq_len = 256
    vocab_size = 4096
    vocab_source = 'custom'
    device = 'cpu'  # 使用 CPU 进行测试
    
    try:
        # 准备数据集参数
        dataset_kwargs = {
            'split': 'train',
            'max_seq_len': max_seq_len,
            'vocab_size': vocab_size,
            'vocab_source': vocab_source
        }
        
        print("1. 测试 Task.iter_batch 方法...")
        # 获取批次迭代器
        batch_iterator = Task.iter_batch(
            batch_size=batch_size,
            device=device,
            num_workers=0,  # 不使用多进程
            **dataset_kwargs
        )
        
        # 获取第一个批次
        first_x, first_y = next(batch_iterator)
        print(f"✓ 成功获取第一个批次!")
        print(f"输入批次形状: {first_x.shape}")
        print(f"标签批次形状: {first_y.shape}")
        print(f"设备: {first_x.device}")
        
        # 验证批次大小是否正确
        assert first_x.shape[0] == batch_size, f"期望批次大小为 {batch_size}，实际为 {first_x.shape[0]}"
        assert first_x.shape[1] == max_seq_len, f"期望序列长度为 {max_seq_len}，实际为 {first_x.shape[1]}"
        print("✓ 批次大小和序列长度验证通过!")
        
        # 验证数据是否在正确的设备上
        assert first_x.device.type == device, f"期望设备为 {device}，实际为 {first_x.device.type}"
        print(f"✓ 设备验证通过!")
        
        # 获取第二个批次
        second_x, second_y = next(batch_iterator)
        print("✓ 成功获取第二个批次!")
        
        print("\nTask 类测试全部通过!")
        return True
        
    except Exception as e:
        print(f"Task 类测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """测试边缘情况"""
    print("\n=== 测试边缘情况 ===")
    
    try:
        # 测试不同的 max_seq_len
        print("1. 测试不同的 max_seq_len...")
        for seq_len in [64, 128, 512]:
            dataset = PretokDataset(
                split='test',  # 使用测试集以减少数据量
                max_seq_len=seq_len,
                vocab_size=4096,
                vocab_source='custom'
            )
            x, y = next(iter(dataset))
            assert x.shape[0] == seq_len, f"对于 seq_len={seq_len}，期望序列长度为 {seq_len}，实际为 {x.shape[0]}"
            print(f"✓ max_seq_len={seq_len} 测试通过!")
        
        # 测试不同的 batch_size
        print("\n2. 测试不同的 batch_size...")
        for b_size in [1, 2, 8]:
            batch_iterator = Task.iter_batch(
                batch_size=b_size,
                device='cpu',
                split='test',
                max_seq_len=64,
                vocab_size=4096,
                vocab_source='custom'
            )
            x, y = next(batch_iterator)
            assert x.shape[0] == b_size, f"对于 batch_size={b_size}，期望批次大小为 {b_size}，实际为 {x.shape[0]}"
            print(f"✓ batch_size={b_size} 测试通过!")
        
        print("\n边缘情况测试全部通过!")
        return True
        
    except Exception as e:
        print(f"边缘情况测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n========== 运行 PretokDataset 和 Task 测试 ==========")
    
    tests = [
        ("PretokDataset 测试", test_pretok_dataset),
        ("Task 测试", test_task),
        ("边缘情况测试", test_edge_cases)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if not test_func():
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试全部通过！")
    else:
        print("\n❌ 部分测试失败！")


if __name__ == "__main__":
    run_all_tests()