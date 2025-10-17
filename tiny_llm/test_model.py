import os
import sys
import torch

# 确保能导入 model 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import Transformer, ModelArgs
from tokenizer import Tokenizer


def test_generate_function():
    """测试 Transformer 类的 generate 函数"""
    print("=== 测试 Transformer.generate 函数 ===")
    
    try:
        # 初始化模型参数
        args = ModelArgs(
            dim=128,          # 减小维度以加快测试速度
            n_layers=2,       # 减少层数
            n_heads=16,        # 减少头数
            n_kv_heads=8,
            vocab_size=4096,  # 使用较小的词汇表
            max_seq_len=64    # 减小最大序列长度
        )
        
        # 初始化模型
        print("1. 初始化模型...")
        model = Transformer(args)
        print(f"✓ 模型初始化成功! 参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 初始化分词器
        print("\n2. 初始化分词器...")
        tokenizer = Tokenizer("data/tok4096.model")
        print(f"✓ 分词器初始化成功! 词汇表大小: {tokenizer.n_words}")
        
        # 准备输入文本和token
        print("\n3. 准备测试数据...")
        test_text = "Once upon a time, there was a little girl named"
        test_tokens = tokenizer.encode(test_text, bos=True, eos=False)
        # 转换为tensor并添加批次维度
        input_tensor = torch.tensor([test_tokens], dtype=torch.long)
        print(f"✓ 输入文本: '{test_text}'")
        print(f"✓ 输入token数量: {len(test_tokens)}")
        
        # 测试基本生成功能
        print("\n4. 测试基本生成功能...")
        max_new_tokens = 10
        generated_tokens = model.generate(input_tensor, max_new_tokens)
        print(f"✓ 生成成功! 生成的token形状: {generated_tokens.shape}")
        assert generated_tokens.shape[1] == len(test_tokens) + max_new_tokens, \
            f"期望输出长度为 {len(test_tokens) + max_new_tokens}，实际为 {generated_tokens.shape[1]}"
        print("✓ 输出长度验证通过!")
        
        # 解码生成的文本
        generated_text = tokenizer.decode(generated_tokens[0].tolist())
        print(f"✓ 生成的文本: '{generated_text}'")
        
        # 测试不同的temperature值
        print("\n5. 测试不同的temperature值...")
        temperatures = [0.0, 0.5, 1.0, 2.0]
        
        for temp in temperatures:
            print(f"  测试 temperature={temp}...")
            with torch.no_grad():
                gen_tokens_temp = model.generate(input_tensor, max_new_tokens, temperature=temp)
                gen_text_temp = tokenizer.decode(gen_tokens_temp[0].tolist())
                print(f"    生成结果: '{gen_text_temp}'")
        
        # 测试不同的top_k值
        print("\n6. 测试不同的top_k值...")
        top_k_values = [1, 5, 20, None]
        
        for top_k in top_k_values:
            print(f"  测试 top_k={top_k}...")
            with torch.no_grad():
                gen_tokens_topk = model.generate(input_tensor, max_new_tokens, temperature=0.7, top_k=top_k)
                gen_text_topk = tokenizer.decode(gen_tokens_topk[0].tolist())
                print(f"    生成结果: '{gen_text_topk}'")
        
        # 测试边界情况：序列长度限制
        print("\n7. 测试序列长度限制...")
        # 创建一个接近最大序列长度的输入
        long_input = torch.randint(0, args.vocab_size, (1, args.max_seq_len - 5), dtype=torch.long)
        # 生成足够的token，使其超过最大序列长度
        with torch.no_grad():
            gen_tokens_long = model.generate(long_input, 20)
        print(f"✓ 长序列生成成功! 输入长度: {long_input.shape[1]}, 输出长度: {gen_tokens_long.shape[1]}")
        
        print("\n8. 测试零温度生成（确定性输出）...")
        # 零温度下应该每次生成相同的结果
        with torch.no_grad():
            gen1 = model.generate(input_tensor, max_new_tokens, temperature=0.0)
            gen2 = model.generate(input_tensor, max_new_tokens, temperature=0.0)
        
        # 检查两次生成的结果是否相同
        same_result = torch.allclose(gen1, gen2)
        print(f"✓ 零温度生成结果一致性: {'通过' if same_result else '失败'}")
        
        print("\n🎉 Transformer.generate 函数测试全部通过!")
        return True
        
    except Exception as e:
        print(f"Transformer.generate 函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generate_edge_cases():
    """测试 Transformer.generate 函数的边缘情况"""
    print("\n=== 测试 Transformer.generate 函数的边缘情况 ===")
    
    try:
        # 初始化简化的模型参数以便快速测试
        args = ModelArgs(
            dim=128,          # 更小的维度
            n_layers=1,      # 只有1层
            n_heads=16,    
            n_kv_heads=8,
            vocab_size=4096, 
            max_seq_len=64   
        )
        print("1",args.n_heads)
        print("2",args.n_kv_heads)
        # 初始化模型
        model = Transformer(args)
        
        # 初始化分词器
        tokenizer = Tokenizer("data/tok4096.model")
        
        # 测试1: 空输入（只有BOS token）
        print("\n1. 测试空输入（只有BOS token）...")
        bos_token = tokenizer.bos_id
        bos_input = torch.tensor([[bos_token]], dtype=torch.long)
        with torch.no_grad():
            gen_from_bos = model.generate(bos_input, 10)
        gen_text_bos = tokenizer.decode(gen_from_bos[0].tolist())
        print(f"✓ 从BOS token生成的文本: '{gen_text_bos}'")
        
        # 测试2: 生成大量token
        print("\n2. 测试生成大量token...")
        short_input = torch.randint(0, args.vocab_size, (1, 3), dtype=torch.long)
        large_max_tokens = 50
        with torch.no_grad():
            gen_large = model.generate(short_input, large_max_tokens)
        print(f"✓ 生成大量token成功! 总长度: {gen_large.shape[1]}")
        assert gen_large.shape[1] == 3 + large_max_tokens, \
            f"期望长度为 {3 + large_max_tokens}，实际为 {gen_large.shape[1]}"
        
        # 测试3: 批量输入
        print("\n3. 测试批量输入...")
        batch_input = torch.randint(0, args.vocab_size, (2, 5), dtype=torch.long)
        with torch.no_grad():
            gen_batch = model.generate(batch_input, 5)
        print(f"✓ 批量生成成功! 输出形状: {gen_batch.shape}")
        assert gen_batch.shape[0] == 2, f"期望批次大小为 2，实际为 {gen_batch.shape[0]}"
        
        print("\n🎉 Transformer.generate 函数边缘情况测试全部通过!")
        return True
        
    except Exception as e:
        print(f"Transformer.generate 函数边缘情况测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n========== 运行 Transformer.generate 函数测试 ==========")
       
    tests = [
        ("基本功能测试", test_generate_function),
        ("边缘情况测试", test_generate_edge_cases)
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