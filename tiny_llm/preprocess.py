from concurrent.futures import ProcessPoolExecutor
import glob
import os
from functools import partial
from tokenizer import Tokenizer
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
import random

DATA_CACHE_DIR = 'data'
TOKENZIER_MODEL = './data/tok4096.model'

def process_shard(args, vocab_size, tokenize_model_path):
    shard_id, shard_filename = args
    
    tokenizer = Tokenizer(tokenize_model_path)

    print(shard_filename)
    with open(shard_filename, 'r') as f:
        data = json.load(f)

    all_tokens = []
    
    for example in tqdm(data, position=shard_id):
        text = example["story"]
        text = text.strip()
        tokens = tokenizer.encode(text, bos=True, eos=True)
        all_tokens.append(tokens)

    all_tokens = np.concatenate(all_tokens).astype(np.uint16)
    all_tokens = np.array(all_tokens, dtype=np.uint16)

    if vocab_size == 0:
        tokenized_filename = shard_filename.replace(".json", ".bin")
    else:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        shard_basename = os.path.basename(shard_filename)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)

    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())

    print(f"Saved {tokenized_filename}.")

def pretokenize(vocab_size:int):
    """
    预处理所有分词之后的数据,将数据保存成二进制文件
    """
    
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")

    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    if vocab_size > 0:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)
    
    fun = partial(process_shard, vocab_size=vocab_size, tokenize_model_path=TOKENZIER_MODEL)

    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    
    print("Pretokenize done.")

class PretokDataset(torch.utils.data.IterableDataset):
    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        """
        初始化数据集

        参数：
        split: str, 数据集的分割方式('train' 或 'test')
        max_seq_len: 最大序列长度
        vocab_size: 词汇表大小
        vocab_source: 词汇表来源 ('custom' 或 'llama2')
        """
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

    def __iter__(self):
        """
        返回迭代器
        """

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Create a PretokDataset with rng seed {seed}")

        if self.vocab_source ==  'llama2':
            bin_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        elif self.vocab_source == 'custom':
            bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        
        # 训练数据集使用除了第一个分片， 测试数据集使用第一个分片
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        assert len(shard_filenames) > 0, f"在 {bin_dir}中没有找到任何.bin文件"

        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len 
                num_batches -= 1
                assert num_batches > 0, "这个分片太小，请检查分布数据"
                idx = list(range(num_batches))
                rng.shuffle(idx)
                for ix in idx:
                    start = ix * self.max_seq_len
                    end =  start + self.max_seq_len + 1
                    chunk = torch.from_numpy(m[start:end].astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


class Task:
    @staticmethod
    def iter_batch(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x,y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x,y 


if __name__ == "__main__":
    pretokenize(vocab_size=4096)