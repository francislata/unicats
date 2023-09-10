from unicats.data.utils import _one_context_config

import torch
import unittest

class TestSliceAudio(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)
    
    def test_slice_audio_one_context_short_audio(self):
        with self.assertRaises(ValueError):
            au = torch.randn((4, 1, 11264))
            _one_context_config(au, 88, 256)

    def test_slice_audio_one_context(self):
        pass


if __name__ == "__main__":
    unittest.main()
