import torch
import ray
import os
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from train_model import train
from functools import partial

runtime_env = {
    "working_dir": "/home/ray/translator-transformer",
    "excludes": ["*.pyc", "__pycache__"],
    "env_vars": {
        "PYTHONPATH": "/home/ray/translator-transformer:$PYTHONPATH"
    }
}

ray.init(address="auto", runtime_env=runtime_env)
scaling_config = ScalingConfig(num_workers=3, use_gpu=False, worker_class_path="/home/ray/translator-transformer")

trainer = TorchTrainer(partial(train, use_ray=True), scaling_config=scaling_config)
trainer.fit()

ray.kill()