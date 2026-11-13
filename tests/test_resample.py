import unittest

from resample import resample_chunks


class ResampleTests(unittest.TestCase):
    def test_single_channel_downsample(self):
        result = resample_chunks([[0, 10, 20, 30]], 4, 2)
        self.assertEqual(result["samples"], [0, 20])
        self.assertEqual(result["frames"], 2)

    def test_empty_chunks(self):
        self.assertEqual(resample_chunks([[]], 4, 2)["samples"], [])


if __name__ == "__main__":
    unittest.main()
