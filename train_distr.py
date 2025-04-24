from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from train_model import train
from functools import partial

scaling_config = ScalingConfig(num_workers=2, use_gpu=False)

trainer = TorchTrainer(partial(train, use_ray=True), scaling_config=scaling_config)
trainer.fit()
