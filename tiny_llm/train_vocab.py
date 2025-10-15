'''
1. 下载数据集
2. 加载数据集到内存 
3. 训练分词器
'''

import argparse
import os
import requests
from tqdm import tqdm
import glob
import json
import sentencepiece

DATA_CACHE_DIR = 'data'
DATASET_NAME = 'TinyStories_all_data'

def download_file(url:str, file_name:str, chunk_size=1024):
    "发送http请求以流式的方式获取url中的数据集"
    resp = requests.get(url, stream=True)

    total = int(resp.headers.get('content-length', 0))

    "从http get请求中获取数据, 并写入文件"
    with open(file_name, "wb") as file, tqdm(
        desc=file_name,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

    
def download_dataset(url:str):
    """
    从url中下载数据集
    """
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    data_filename = os.path.join(DATA_CACHE_DIR, url.split('/')[-1])

    if not os.path.exists(data_filename):
        download_file(url, data_filename)
    else:
        print(f"{data_filename} alreay exists, skipping download")

    data_dir = os.path.join(DATA_CACHE_DIR, url.split('/')[-1].split('.')[0])

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unzipping {data_filename}")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"f{data_dir} alreay exists, skipping unzipping")

    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    with open(shard_filenames[0], 'r') as f:
        data = json.load(f)

    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example story:\n{data[0]}")

def train_vocab(vocab_size:int, num_shard=20):
    """
    vocab_size: 训练词汇表的大小
    num_shard: 指定每个数据块用来训练词汇表的分片大小, 用于加速词汇表训练
    """

    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")

    
    #加载部分数据集到临时文件用于训练
    tiny_file = os.path.join(DATA_CACHE_DIR, 'tiny.txt')
    data_dir = os.path.join(DATA_CACHE_DIR, DATASET_NAME)
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, '*.json')))

    print(f"Writing temporary file {tiny_file} with {num_shard} shard..")
    with open(tiny_file, "w", encoding="utf-8") as of:
        for shard_file in shard_filenames[:num_shard]:
            with open(shard_file, 'r') as f:
                data = json.load(f)
                for example in data:
                    text = example["story"]
                    text = text.strip()
                    of.write(text + '\n')

    print(f"tiny.txt size is {os.path.getsize(tiny_file) / 1024 / 1024} MB.")

    #开始训练分词器
    print("Now train the vocab...")
    sentencepiece.SentencePieceTrainer.train(
        input=tiny_file,
        model_prefix=prefix,
        model_type='bpe',
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format='text',
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r" \332\201\207",
        normalization_rule_name='identity'
    )

    ### 是否删除临时文件tiny.txt
    dec = input(f"Delte the temporary file {tiny_file}? [y/N]")
    if dec.lower() == 'y':
        os.remove(tiny_file)
        print(f"Deleted {tiny_file}")
    
    ### 输出模型报保存的路径
    print(f"Trained tokenzier is in {prefix}.model")
    print("Tokenzier train done.")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", default=False)
    parser.add_argument("--url", type=str, default=
        "https://www.modelscope.cn/datasets/AI-ModelScope/TinyStories/resolve/master/TinyStories_all_data.tar.gz")
    parser.add_argument("--vocab_size", type=int, default=32000)
    args = parser.parse_args()
    if args.download:
        download_dataset(args.url)

    train_vocab(args.vocab_size)


