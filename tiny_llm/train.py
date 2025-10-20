import torch
from model import Transformer, ModelArgs
from preprocess import Task
from functools import partial
import time
import os
from contextlib import nullcontext
import math
#-------------------------------------
# I/0 配置， 用于定义输出目录和训练时的日志配置与评估设置
output_dir = "output"  ### 模式输出保存路径
eval_interval= 2000   ### 评估间隔步数
eval_iters = 100 #每次评估时的迭代步数， 取多次迭代平均
eval_only = False
always_save_checkpoints = False # 每次评估之后总是保存检查点
init_from = "scratch" # 从头训练("scratch") 或者 从已有的检查点恢复 ("resume")

# 训练数据参数
batch_size = 8
max_seq_len = 100
vocab_size = 4096
vocab_source = "custom"

# 模型参数
dim = 512
n_layers= 8
n_heads = 8
n_kv_heads = 4
multiple_of = 32
dropout = 0

#优化器配置
gradient_accumulation_steps = 4
learning_rate = 5e-4
max_iters = 100000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

#学习率配置
decay_lr = True
warmup_iters = 1000

#系统配置
device = "cuda"
dtype = "bfloat16"

lr_decay_iter = max_iters
min_lr = 0.0
master_process=True


#--------------------------------------
#获取配置参数的键值对，便于后续的日志记录
config_keys = [
    k
    for k,v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
config = {k:globals()[k] for k in config_keys}
print(config)
seed_offset = 0 #随机种子偏移量
##设置随机种子，确保可重复性
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = torch.float16

ctx = (
    nullcontext()
    if device_type == "cpu" else torch.amp.autocast(device_type=device_type,dtype=ptdtype)
)

iter_batches = partial(
    Task.iter_batch,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    vocab_size=vocab_size,
    vocab_source=vocab_source,
    device=device,
    num_workers=0,
)

## 训练迭代次次数初始化
iter_num = 0

# 验证集上最好的损失初始化设置为一个极大值，用于后续模型验证时对比更新
best_val_loss = 1e9

#模型参数
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout
)

#-----------------------------------------
#模型初始化
gptconf = ModelArgs(**model_args)
model=Transformer(gptconf)

model.to(device)

#初始化GradScaler， 用于自动混合精度训练
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

#优化器初始化
optimizer = model.configure_optimizers(
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    betas = (beta1, beta2),
    device_type=device_type
)

#定义评估损失的函数
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
            losses[k] = loss.item()
        out[split]=losses.mean()
    model.train()
    return out


#定义学习率调度函数
def get_lr(it):
    """
    根据当前的训练迭代的步数iter返回当前的学习率值。
    学习率调整策略包括线性预热、余弦退火和最小学习率限制。
    """
    #1) 线性预热， 学习率线性增长
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    #2) 如果迭代步数超过 lr_decay_iter, 返回最小学习率
    if it > lr_decay_iter:
        return min_lr

    #3) 余弦退火阶段，在 warmup_iter 和 lr_decay_iter
    decay_ratio = (it - warmup_iters) / (lr_decay_iter - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr) 

### 初始化训练数据的迭代器
train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)
t0 = time.time()
local_iter_num = 0
raw_model = model
running_mfu = -1.0

os.makedirs(output_dir, exist_ok=True)

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses["val"] < best_val_loss or always_save_checkpoints:
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model":raw_model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "model_args":model_args,
                    "iter_num":iter_num,
                    "best_val_loss":best_val_loss,
                    "config":config
                }
                print(f"saving checkpint to {output_dir}")
                torch.save(checkpoint, os.path.join(output_dir, "ckpt.pt"))

    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits = model(X, Y)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps
        X, Y  = next(train_batch_iter)
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    iter_num += 1
    
    if iter_num > max_iters:
        break
        