from typing import Tuple

import torch

def one_context_config(audio: torch.Tensor, context_length: int, hop_size: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    audio_length = context_length * hop_size
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

def two_context_config(audio: torch.Tensor, min_input_frames: int, hop_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    min_input_length = min_input_frames * hop_size

    if audio.size(-1) < min_input_length:
        raise ValueError("audio length is shorter than minimum input length")

    input_length = torch.randint(min_input_length, audio.size(-1) - min_input_length, (1,)).item()
    input_start_idx = torch.randint(0, audio.size(-1) - input_length, (1,)).item()
    input_audio = audio[..., input_start_idx:input_start_idx + input_length]
    a_context_audio = audio[..., :input_start_idx]
    b_context_audio = audio[..., input_start_idx + input_length:]

    return (
        a_context_audio,
        input_audio,
        b_context_audio,
        input_start_idx
    )
    
def collate_fn(batch):
    # TODO: Ensure batch is of equal length to the longest sequence.
    # Afterwards, randomly select which config to use (might need to modify slice_audio)
    # and then also pass in the mask to use.
    for sample in batch:
        print(sample.shape)