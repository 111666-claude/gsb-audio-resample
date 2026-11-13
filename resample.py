from __future__ import annotations


def _frames(samples, channels):
    return [samples[index:index + channels] for index in range(0, len(samples), channels)]


def resample_chunks(chunks: list[list[int]], src_rate: int, dst_rate: int, channels: int = 1) -> dict:
    step = float(src_rate) / float(dst_rate)
    output = []
    for chunk in chunks:
        frames = _frames(chunk, channels)
        if not frames:
            continue
        position = 0.0
        while position < len(frames) - 1:
            left = frames[int(position)]
            right = frames[min(int(position) + 1, len(frames) - 1)]
            fraction = position - int(position)
            for channel in range(channels):
                value = left[channel] + (right[channel] - left[channel]) * fraction
                output.append(int(round(value)))
            position += step
    return {"samples": output, "frames": len(output) // channels, "work": sum(len(chunk) // channels for chunk in chunks)}
