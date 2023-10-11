from glob import glob
from typing import Dict, Any, Optional
from unicats.data.utils import collate_fn

import librosa
import numpy as np
import os
import torch.utils.data as data


class LibriTTSDataset(data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        hop_size: int,
        num_frames: int,
        au_format: str = "wav",
        sample_rate: Optional[int] = None
    ):
        self.utterances = glob(os.path.join(dataset_path, f"**/*.{au_format}"), recursive=True)
        self.hop_size = hop_size
        self.num_frames = num_frames
        self.sample_rate = sample_rate

    def __getitem__(self, index: int) -> np.ndarray:
        utterance = self.utterances[index]
        au, _ = librosa.load(utterance, sr=self.sample_rate)
        # TODO: Apply tokenizer here.
        return au

    def __len__(self) -> int:
        return len(self.utterances)