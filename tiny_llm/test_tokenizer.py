import os
import sys

# 确保能导入tokenizer模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import Tokenzier

def test_tokenizer():
    # 测试1: 初始化分词器
    print("=== 测试1: 初始化分词器 ===")
    try:
        tokenizer = Tokenzier()
        print("分词器初始化成功!")
        print(f"词汇表大小: {tokenizer.n_words}")
        print(f"BOS ID: {tokenizer.bos_id}")
        print(f"EOS ID: {tokenizer.eos_id}")
        print(f"PAD ID: {tokenizer.pad_id}")
    except Exception as e:
        print(f"分词器初始化失败: {e}")
        return False
    
    # 测试2: 基本的编码解码功能
    print("\n=== 测试2: 基本的编码解码功能 ===")
    test_text = "Hello, world! This is a test."
    try:
        # 不带BOS和EOS的编码
        tokens = tokenizer.encode(test_text, bos=False, eos=False)
        print(f"原始文本: {test_text}")
        print(f"编码结果: {tokens}")
        print(f"Token数量: {len(tokens)}")
        
        # 解码回文本
        decoded_text = tokenizer.decode(tokens)
        print(f"解码结果: {decoded_text}")
        
        # 验证编码解码的一致性
        assert decoded_text == test_text, f"编码解码不一致: 期望 '{test_text}', 得到 '{decoded_text}'"
        print("✓ 编码解码一致性验证通过!")
    except Exception as e:
        print(f"基本编码解码测试失败: {e}")
        return False
    
    # 测试3: 带BOS和EOS的编码
    print("\n=== 测试3: 带BOS和EOS的编码 ===")
    try:
        tokens_with_bos_eos = tokenizer.encode(test_text, bos=True, eos=True)
        print(f"带BOS和EOS的编码结果: {tokens_with_bos_eos}")
        print(f"Token数量: {len(tokens_with_bos_eos)}")
        
        # 验证BOS和EOS是否正确添加
        assert tokens_with_bos_eos[0] == tokenizer.bos_id, "BOS ID未正确添加"
        assert tokens_with_bos_eos[-1] == tokenizer.eos_id, "EOS ID未正确添加"
        assert tokens_with_bos_eos[1:-1] == tokens, "中间的token与不带BOS/EOS的结果不一致"
        print("✓ BOS和EOS添加验证通过!")
        
        # 解码带BOS和EOS的文本
        decoded_with_bos_eos = tokenizer.decode(tokens_with_bos_eos)
        print(f"带BOS和EOS的解码结果: '{decoded_with_bos_eos}'")
    except Exception as e:
        print(f"带BOS和EOS的编码测试失败: {e}")
        return False
    
    # 测试4: 特殊字符和长文本
    print("\n=== 测试4: 特殊字符和长文本 ===")
    special_text = "你好，世界！12345 😊 中文测试 英文测试 Special chars: !@#$%^&*()"
    long_text = "This is a longer text that tests how well the tokenizer handles multiple sentences. It includes various words and punctuation. Let's see how it performs." * 3
    
    try:
        # 测试特殊字符
        special_tokens = tokenizer.encode(special_text, bos=False, eos=False)
        special_decoded = tokenizer.decode(special_tokens)
        print(f"特殊字符文本: {special_text}")
        print(f"特殊字符解码结果: {special_decoded}")
        print(f"特殊字符Token数量: {len(special_tokens)}")
        
        # 测试长文本
        long_tokens = tokenizer.encode(long_text, bos=False, eos=False)
        long_decoded = tokenizer.decode(long_tokens)
        print(f"长文本Token数量: {len(long_tokens)}")
        print(f"长文本解码结果长度: {len(long_decoded)}")
        print(f"长文本解码前几个字符: {long_decoded[:50]}...")
        
        print("✓ 特殊字符和长文本测试通过!")
    except Exception as e:
        print(f"特殊字符和长文本测试失败: {e}")
        return False
    
    # 测试5: 空字符串处理
    print("\n=== 测试5: 空字符串处理 ===")
    try:
        empty_tokens = tokenizer.encode("", bos=True, eos=True)
        print(f"空字符串带BOS和EOS的编码: {empty_tokens}")
        assert len(empty_tokens) == 2, f"空字符串带BOS和EOS应该只有2个token，实际有{len(empty_tokens)}个"
        assert empty_tokens[0] == tokenizer.bos_id and empty_tokens[1] == tokenizer.eos_id, "空字符串的BOS/EOS不正确"
        
        empty_decoded = tokenizer.decode(empty_tokens)
        print(f"空字符串解码结果: '{empty_decoded}'")
        print("✓ 空字符串处理测试通过!")
    except Exception as e:
        print(f"空字符串处理测试失败: {e}")
        return False
    
    print("\n所有测试通过！")
    return True

if __name__ == "__main__":
    test_tokenizer()