"""Wall clock that AppWorld's freezegun patching cannot alter (freezegun rewrites time.time/monotonic/perf_counter and
module-level references to them; os.times().elapsed comes straight from the times() syscall)."""
import os


def wall() -> float:
    return os.times().elapsed
