import tensorflow as tf
import numpy as np
from PIL import Image
from src.data.dataset import prepare_b2d_dataset
from src.utils.monitor import log_execution_time
from torch.utils.data import DataLoader


tf.config.set_visible_devices([], "GPU")

class Bench2DriveDataset:
    @log_execution_time()
    def __init__(self, config):
        dataset = prepare_b2d_dataset(
            work_dir=config.work_dir,
            statistics_path=config.statistics_path,
            use_fixed_images=config.use_fixed_images,
            num_of_action_experts=config.num_of_action_experts,
            split=config.get("split", None)
        )
        self.dataset = dataset


if __name__ == '__main__':
    from omegaconf import OmegaConf
    base_config = OmegaConf.load("config/train/DrivePi0/base.yaml")
    base_dataset = Bench2DriveDataset(base_config.data.train).dataset
    test_dataloader = DataLoader(base_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    for batch in test_dataloader:
        print('state_shape: ', batch['state'].shape)
        print('trajectory_shape: ', batch['trajectory'].shape)
        break