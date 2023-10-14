from glob import glob
from typing import Optional, Tuple
from unicats.data.utils import collate_fn

import librosa
import numpy as np
import os
import torch.utils.data as data


class LibriTTSDataset(data.Dataset):
    def __init__(
        self,
        utterance_config_file_path: str,
        hop_size: int,
        num_frames: int,
        sample_rate: Optional[int] = None
    ):
        self.utterances = []
        self.context_configs = []

        with open(utterance_config_file_path) as utterance_config_fp:
            for line in utterance_config_fp.readlines():
                utterance_file_path, context_config = line.strip().split(",")

                self.utterances.append(utterance_file_path)
                self.context_configs.append(context_config)

        self.hop_size = hop_size
        self.num_frames = num_frames
        self.sample_rate = sample_rate

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        utterance = self.utterances[index]
        context_config = self.context_configs[index]

        au, _ = librosa.load(utterance, sr=self.sample_rate)
        # TODO: Apply tokenizer here.
        # TODO: Apply context helpers here based on `context_config`
        return (au, context_config)

    def __len__(self) -> int:
        return len(self.utterances)