import torch
import ray
import os
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from train_model import train
from functools import partial

os.environ['PYTHONPATH'] = '/home/ray/translator-transformer'

ray.init(address="auto")
scaling_config = ScalingConfig(num_workers=3, use_gpu=False)

trainer = TorchTrainer(partial(train, use_ray=True), scaling_config=scaling_config)
trainer.fit()

ray.kill()