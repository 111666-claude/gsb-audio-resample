import random
import unittest

from resample import StreamResampler, resample_chunks


def _flat(chunks):
    return [sample for chunk in chunks for sample in chunk]


class ResampleTests(unittest.TestCase):
    def test_single_channel_downsample(self):
        result = resample_chunks([[0, 10, 20, 30]], 4, 2)
        self.assertEqual(result["samples"], [0, 20])
        self.assertEqual(result["frames"], 2)

    def test_empty_chunks(self):
        self.assertEqual(resample_chunks([[]], 4, 2)["samples"], [])

    def test_split_matches_contiguous(self):
        split = resample_chunks([[0, 10], [20, 30, 40]], 3, 2)
        joined = resample_chunks([[0, 10, 20, 30, 40]], 3, 2)
        self.assertEqual(split, joined)
        self.assertEqual(joined["samples"], [0, 15, 30])
        self.assertEqual(joined["frames"], 3)
        self.assertEqual(joined["work"], 5)

    def test_empty_chunks_do_not_reset_phase(self):
        baseline = resample_chunks([[0, 10, 20, 30, 40]], 3, 2)
        noisy = resample_chunks([[], [0, 10], [], [], [20, 30, 40], []], 3, 2)
        self.assertEqual(noisy, baseline)

    def test_stereo_interpolates_per_frame(self):
        result = resample_chunks([[0, 100, 10, 110, 20, 120]], 3, 2, channels=2)
        self.assertEqual(result["samples"], [0, 100, 15, 115])
        self.assertEqual(result["frames"], 2)

    def test_stereo_split_mid_frame(self):
        joined = resample_chunks([[0, 100, 10, 110, 20, 120]], 3, 2, channels=2)
        split = resample_chunks([[0, 100, 10], [110, 20, 120]], 3, 2, channels=2)
        self.assertEqual(split, joined)

    def test_upsampling_and_flush(self):
        result = resample_chunks([[0, 10]], 2, 4)
        self.assertEqual(result["samples"], [0, 5, 10])
        self.assertEqual(result["frames"], 3)
        self.assertEqual(result["work"], 2)

    def test_work_counts_input_frames_once(self):
        for rates in [(3, 2), (2, 3), (4, 2), (5, 7), (7, 5)]:
            src, dst = rates
            data = list(range(24))
            joined = resample_chunks([data], src, dst)
            split = resample_chunks([data[:5], data[5:6], data[6:17], data[17:]], src, dst)
            self.assertEqual(split, joined)
            self.assertEqual(joined["work"], len(data))

    def test_rounding_is_deterministic(self):
        # 1 -> 2: grid positions 0, 0.5, 1, 1.5, 2.  The 1.5 midpoint must round
        # half to even (1.5 -> 2), and the exact final grid point is flushed.
        result = resample_chunks([[0, 1, 2]], 1, 2)
        self.assertEqual(result["samples"], [0, 0, 1, 2, 2])
        # Same values regardless of where the stream is cut.
        self.assertEqual(
            resample_chunks([[0], [1], [2]], 1, 2)["samples"], result["samples"]
        )

    def test_random_chunkings_match(self):
        rng = random.Random(20261113)
        for _ in range(200):
            src = rng.randint(1, 9)
            dst = rng.randint(1, 9)
            channels = rng.choice([1, 2, 3])
            frames = rng.randint(0, 20)
            samples = [rng.randint(-1000, 1000) for _ in range(frames * channels)]
            joined = resample_chunks([samples], src, dst, channels)
            chunks = []
            cursor = 0
            while cursor < len(samples):
                size = rng.randint(0, 5)
                chunks.append(samples[cursor:cursor + size])
                cursor += size
            if cursor == len(samples):
                chunks.append([])
            split = resample_chunks(chunks, src, dst, channels)
            self.assertEqual(split, joined, (src, dst, channels, frames, chunks))
            self.assertEqual(joined["samples"], resample_chunks([_flat(chunks)], src, dst, channels)["samples"])
            self.assertEqual(joined["work"], frames)
            self.assertEqual(len(joined["samples"]) % channels, 0)

    def test_streaming_api_matches_batch(self):
        stream = StreamResampler(3, 2)
        stream.feed([0, 10])
        stream.feed([])
        stream.feed([20, 30, 40])
        self.assertEqual(stream.finish(), resample_chunks([[0, 10, 20, 30, 40]], 3, 2))


if __name__ == "__main__":
    unittest.main()
