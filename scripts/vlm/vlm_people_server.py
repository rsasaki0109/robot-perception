#!/usr/bin/env python3
"""People in images, one JSON line each way, for a robot's control loop.

Reads `{"image": "/path/frame.png"}` lines on stdin and answers each with
`{"people": [[x0, y0, x1, y1], ...], "seconds": s}` (pixels) on stdout.
Florence-2's `<OD>` prompt finds the people, as in `vlm_detect.py`.

    python3 scripts/vlm/vlm_people_server.py --threads 2

Robot Native Engine's example 151 starts it with `--vlm`.
"""

import argparse
import json
import sys
import time

from vlm_detect import PERSON_LABELS, load, prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="florence-community/Florence-2-base")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    from PIL import Image

    processor, model = load(args.model, args.threads, args.dtype)
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        start = time.perf_counter()
        image = Image.open(request["image"]).convert("RGB")
        people = [box for label, box in prompt(processor, model, image, "<OD>") if label in PERSON_LABELS]
        reply = {"people": people, "seconds": round(time.perf_counter() - start, 3)}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
