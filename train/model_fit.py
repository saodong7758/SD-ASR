import os
import torch

from funasr import AutoModel
from funasr.train_utils.train import Trainer

# ==============================
# 路径配置
# ==============================

model_path = "/mnt/d/models/fun-asr-nano-2512"

train_file = "/mnt/d/Fun-ASR/data/train_nano.jsonl"

valid_file = "/mnt/d/Fun-ASR/data/dev_nano.jsonl"

output_dir = "/mnt/d/Fun-ASR/output_shanghai_model"


os.makedirs(output_dir, exist_ok=True)


# ==============================
# 加载模型
# ==============================

print("正在加载 Fun-ASR-Nano...")

model = AutoModel(
    model=model_path,
    trust_remote_code=True,
    disable_update=True
)

print("模型加载完成")


# ==============================
# LoRA配置
# ==============================

train_config = {

    # 数据
    "train_data": train_file,
    "valid_data": valid_file,


    # 输出
    "output_dir": output_dir,


    # 训练参数
    "max_epoch": 10,

    "batch_size": 2,

    "gradient_accumulation_steps": 8,

    "learning_rate": 1e-4,


    # 保存
    "save_checkpoint_interval": 1000,


    # LoRA
    "use_lora": True,

    "lora_rank": 8,

    "lora_alpha": 16,


    # 混合精度
    "fp16": True,


    # GPU
    "device": "cuda"

}


# ==============================
# 开始训练
# ==============================

print("开始上海话LoRA微调...")


trainer = Trainer(
    model=model,
    **train_config
)


trainer.train()


print("====================")
print("训练完成")
print(output_dir)