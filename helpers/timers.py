from typing import Optional

import time
import wandb


class SegmentTimer:
    """
    Timer for measuring elapsed time of code segments.
    """
    def __init__(self) -> None:
        """
        Initialize the timer and prepare segment storage.
        """
        self._start: float = time.perf_counter()
        self._last: float = self._start
        self._segments: List[Tuple[str, float]] = []

    def segment(self, name: str, print_time: bool = True) -> float:
        """
        Record a new segment.

        Args:
            name (str): Description of this segment.
            print_time (bool): If True, print the elapsed time for this segment.

        Returns:
            float: Time elapsed since the previous segment (in seconds).
        """
        now: float = time.perf_counter()
        elapsed: float = now - self._last
        self._segments.append((name, elapsed))
        self._last = now
        if print_time:
            print(f"[Segment] {name}: {elapsed:.4f}s")
        return elapsed

    def print_segments(self) -> None:
        """
        Print all recorded segments with their names and elapsed times,
        plus the total time since initialization.
        """
        total: float = time.perf_counter() - self._start
        print(f"{'Segment':<30}{'Time (s)':>10}")
        print("-" * 40)
        for name, elapsed in self._segments:
            print(f"{name:<30}{elapsed:>10.4f}")
        print("-" * 40)
        print(f"{'Total':<30}{total:>10.4f}")

    def log_to_wandb(self, step: Optional[int] = None) -> None:
        """
        Log segments to W&B with minimal metric keys (segment names and 'total').

        Args:
            step: Optional W&B step index.
        """
        metrics = {name + "_time": elapsed for name, elapsed in self._segments}
        metrics["total_time"] = time.perf_counter() - self._start
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)


if __name__ == '__main__':
    timer = SegmentTimer()

    # Segment A
    # (e.g., data loading)
    # ... your code here ...
    timer.segment("load_data")

    # Segment B
    # (e.g., model training)
    # ... your code here ...
    timer.segment("train_model")

    # Segment C
    # (e.g., evaluation)
    # ... your code here ...
    timer.segment("evaluate")

    # Print results to console
    timer.print_segments()