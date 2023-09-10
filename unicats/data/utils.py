import torch

def _one_context_config(audio: torch.Tensor, context_length: int, hop_length: int):
    audio_length = context_length * hop_length
    sequence_length = torch.randint(2, 4, (1,)).item()

    if audio.size(-1) < audio_length * sequence_length:
        raise ValueError("audio length is shorter than context length")

    return (audio[..., :(audio_length * sequence_length)], audio[..., (audio_length * sequence_length):])


def slice_audio(au: torch.Tensor):
    # TODO:
    # 1) Define which configuration is used randomly. Perform some weighted random probability.
    # 2) Once chosen, call that config.
    pass
