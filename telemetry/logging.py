from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timed(stage: str):
    t0 = time.time()
    try:
        yield
    finally:
        dt = int((time.time() - t0) * 1000)
        print(f"{stage}: {dt} ms")
