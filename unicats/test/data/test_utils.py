from unicats.data.utils import _one_context_config

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
            _one_context_config(audio, self.context_length, self.hop_length)

    def test_slice_audio_one_context(self):
        audio = torch.randn((4, 1, 135168))

        context_audio, target_audio, context_end_length = _one_context_config(
            audio,
            self.context_length,
            self.hop_length
        )

        torch.testing.assert_close(context_audio, audio[..., :context_end_length])
        torch.testing.assert_close(target_audio, audio[..., context_end_length:])


if __name__ == "__main__":
    unittest.main()
