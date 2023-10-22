from unicats.data.utils import one_context_config, two_context_config

import torch
import unittest

class TestSliceAudio(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)

        self.context_length = 88
        self.hop_length = 256
    
    def test_slice_audio_one_context_short_audio(self):
        audio = torch.randn((4, 1, 11264))

        with self.assertRaises(ValueError):
            one_context_config(audio, self.context_length, self.hop_length)

    def test_slice_audio_one_context(self):
        audio = torch.randn((4, 1, 135168))

        context_audio, target_audio, context_end_length = one_context_config(
            audio,
            self.context_length,
            self.hop_length
        )

        torch.testing.assert_close(context_audio, audio[..., :context_end_length])
        torch.testing.assert_close(target_audio, audio[..., context_end_length:])

    def test_slice_audio_two_context_short_audio(self):
        audio = torch.randn((4, 1, 11264))

        with self.assertRaises(ValueError):
            two_context_config(audio, self.context_length, self.hop_length)

    def test_slice_audio_two_context(self):
        audio = torch.randn((4, 1, 135168))

        a_context_audio, input_audio, b_context_audio, input_start_idx = two_context_config(
            audio,
            self.context_length,
            self.hop_length
        )

        torch.testing.assert_close(a_context_audio, audio[..., :input_start_idx])
        torch.testing.assert_close(input_audio, audio[..., input_start_idx:input_start_idx + input_audio.size(-1)])
        torch.testing.assert_close(b_context_audio, audio[..., input_start_idx + input_audio.size(-1):])


if __name__ == "__main__":
    unittest.main()
