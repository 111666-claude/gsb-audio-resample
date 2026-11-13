from __future__ import annotations


def _frames(samples, channels):
    return [samples[index:index + channels] for index in range(0, len(samples), channels)]


def _round_div(num: int, den: int) -> int:
    """Round ``num / den`` to the nearest integer, ties to even.

    Uses integer arithmetic only, so the result is deterministic for every
    rate pair (and matches Python's ``round()`` on the exact rational value).
    """
    quotient, remainder = divmod(num, den)
    doubled = remainder * 2
    if doubled > den or (doubled == den and quotient % 2 != 0):
        quotient += 1
    return quotient


class StreamResampler:
    """Linear-interpolation resampler whose phase survives chunk boundaries.

    The cursor is kept as an integer numerator in units of ``1 / dst_rate``
    source frames.  This makes the sampling grid exact and monotonic no matter
    how the input stream is split, and avoids floating point drift.
    """

    def __init__(self, src_rate: int, dst_rate: int, channels: int = 1):
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError("sample rates must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.src_rate = int(src_rate)
        self.dst_rate = int(dst_rate)
        self.channels = int(channels)

        self._phase = 0           # cursor, in units of 1 / dst_rate frames
        self._buffer = []         # frames not yet consumed, from _buffer_start
        self._buffer_start = 0    # absolute index of _buffer[0]
        self._pending = []        # raw samples of an incomplete trailing frame
        self._total_frames = 0    # complete input frames seen so far
        self._consumed = 0        # input frames consumed
        self._output = []

    def feed(self, samples) -> None:
        """Append interleaved samples; emits every frame that is now decidable."""
        self._pending.extend(samples)
        usable = len(self._pending) - len(self._pending) % self.channels
        frames = _frames(self._pending[:usable], self.channels)
        self._pending = self._pending[usable:]
        if not frames:
            return
        self._buffer.extend(frames)
        self._total_frames += len(frames)
        self._consumed += len(frames)
        self._drain()

    def finish(self) -> dict:
        """Flush the stream and return the same JSON-shaped result as before."""
        self._drain()
        return {
            "samples": self._output,
            "frames": len(self._output) // self.channels,
            "work": self._consumed,
        }

    def _drain(self) -> None:
        while True:
            index = self._phase // self.dst_rate
            remainder = self._phase % self.dst_rate
            if index >= self._total_frames:
                break

            local = index - self._buffer_start
            if index + 1 < self._total_frames:
                left = self._buffer[local]
                right = self._buffer[local + 1]
            elif remainder == 0:
                # Cursor sits exactly on the last known frame.  Interpolating
                # at fraction 0 yields that frame itself, so the value is final
                # and can be emitted without waiting for more input.
                left = right = self._buffer[local]
            else:
                # Fractional position past the last known frame: wait for more
                # input; during flush this is the natural end of the stream.
                break

            for channel in range(self.channels):
                numerator = (
                    left[channel] * self.dst_rate
                    + (right[channel] - left[channel]) * remainder
                )
                self._output.append(_round_div(numerator, self.dst_rate))

            self._phase += self.src_rate

        self._trim()

    def _trim(self) -> None:
        """Drop frames the cursor can no longer reference (each frame once)."""
        keep = self._phase // self.dst_rate
        drop = keep - self._buffer_start
        if drop <= 0:
            return
        if drop >= len(self._buffer):
            self._buffer_start += len(self._buffer)
            self._buffer.clear()
        else:
            del self._buffer[:drop]
            self._buffer_start += drop


def resample_chunks(chunks: list[list[int]], src_rate: int, dst_rate: int, channels: int = 1) -> dict:
    resampler = StreamResampler(src_rate, dst_rate, channels)
    for chunk in chunks:
        resampler.feed(chunk)
    return resampler.finish()
