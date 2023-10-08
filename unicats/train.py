import hydra
import torch.utils.data as data
import os

from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any

from unicats.data.dataset import LibriTTSDataset


def _setup_dataloader(dataset_config: Dict[str, Any]) -> data.DataLoader:
    dataset = LibriTTSDataset(
        os.path.join(dataset_config["dataset_dir"], "train-other-500"),
        dataset_config["dataloader"],
        dataset_config["hop_size"],
        dataset_config["num_frames"],
        sample_rate=dataset_config["sample_rate"]
    )

    return dataset.dataloader


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    config = OmegaConf.to_object(config)

    dataloader = _setup_dataloader(config["dataset"])

    for batch in dataloader:
        break


if __name__ == "__main__":
    main()