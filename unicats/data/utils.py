from enum import Enum

import numpy as np
import random
import torch


def one_context_config(au: torch.Tensor, hop_size: int, sampling_rate: int) -> tuple[torch.Tensor, torch.Tensor]:
    ctx_min_frame_length = int(2 * sampling_rate / hop_size)
    ctx_max_frame_length = int(3 * sampling_rate / hop_size)
    ctx_frame_length = random.randint(ctx_min_frame_length, ctx_max_frame_length)

    return (
        au[:ctx_frame_length],
        au[ctx_frame_length:]
    )

def two_context_config(au: torch.Tensor, min_input_frames: int, hop_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_max_frame_length = (au.shape[-1] // hop_size) - min_input_frames
    input_frame_length = random.randint(min_input_frames, input_max_frame_length)
    input_start_frame = random.randint(0, input_max_frame_length)

    return (
        au[:input_start_frame * hop_size],
        au[input_start_frame * hop_size:(input_start_frame + input_frame_length) * hop_size],
        au[(input_start_frame + input_frame_length) * hop_size:]
    )


class ContextCollator(object):
    def __init__(
        self,
        hop_size: int = 256,
        win_length: int = 1024,
        sampling_rate: int = 16000,
        n_mel: int = 80,
        context_config_probs: list[float] = [0.6, 0.3, 0.1] # NOTE: prob. order as follows: [<two context config prob>, <one context config prob>, <no context config prob>]
    ):
        self.hop_size = hop_size
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel = n_mel
        self.context_config_probs = torch.Tensor(context_config_probs)

    def __call__(
        self,
        batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> torch.Tensor:
        batch = batch[0]
        ctx_config_indices = [
            ctx_config_idx.item()
            for ctx_config_idx in torch.multinomial(self.context_config_probs, len(batch), replacement=True)
        ]

        aus = []
        for (ctx_config_index, input) in zip(ctx_config_indices, batch):
            if ctx_config_index == 0:
                aus.append(two_context_config(input[0], 100, self.hop_size))
            elif ctx_config_index == 1:
                aus.append(one_context_config(input[0], self.hop_size, self.sampling_rate))
            else:
                aus.append(input[0])

        return None
