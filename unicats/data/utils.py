from typing import Tuple

import torch

def _one_context_config(audio: torch.Tensor, context_length: int, hop_length: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    audio_length = context_length * hop_length
    # TODO: consider moving the sequence length selection to be passed in as arguments
    sequence_length = torch.randint(2, 4, (1,)).item()
    context_end_length = audio_length * sequence_length

    if audio.size(-1) < context_end_length:
        raise ValueError("audio length is shorter than context length")

    return (
        audio[..., :context_end_length],
        audio[..., context_end_length:],
        context_end_length
    )


def slice_audio(au: torch.Tensor):
    # TODO:
    # 1) Define which configuration is used randomly. Perform some weighted random probability.
    # 2) Once chosen, call that config.
    pass
