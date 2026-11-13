import argparse
import json

from resample import resample_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="split")
    args = parser.parse_args()
    if args.sample == "split":
        chunks = [[0, 10], [20, 30, 40]]
        result = resample_chunks(chunks, 3, 2)
    elif args.sample == "stereo":
        result = resample_chunks([[0, 100, 10, 110, 20, 120]], 3, 2, channels=2)
    else:
        raise SystemExit("unknown sample")
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
