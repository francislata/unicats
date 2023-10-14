import hydra
import torch.utils.data as data
import os

from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any

from unicats.data.dataset import LibriTTSDataset
from unicats.data.utils import collate_fn


def _setup_dataloader(dataset_config: Dict[str, Any]) -> data.DataLoader:
    dataset = LibriTTSDataset(
        dataset_config["utterance_config_file_path"],
        dataset_config["hop_size"],
        dataset_config["num_frames"],
        sample_rate=dataset_config["sample_rate"]
    )

    return data.DataLoader(dataset, collate_fn=collate_fn, **dataset_config["dataloader"])


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    config = OmegaConf.to_object(config)

    dataloader = _setup_dataloader(config["dataset"])


if __name__ == "__main__":
    main()