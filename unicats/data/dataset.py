from glob import glob
from typing import Optional

import librosa
import numpy as np
import os
import torch.utils.data as data


class LibriTTSDataset(data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        au_format: str = "wav",
        sample_rate: Optional[int] = None,
    ):
        self.utterances = glob(os.path.join(dataset_path, f"**/*.{au_format}"), recursive=True)
        self.sample_rate = sample_rate

    def __getitem__(self, index: int) -> np.ndarray:
        utterance = self.utterances[index]
        au, _ = librosa.load(utterance, sr=self.sample_rate)
        return au

    def __len__(self) -> int:
        return len(self.utterances)