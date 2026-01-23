import time
from datetime import datetime
from zoneinfo import ZoneInfo

def start_timer() -> tuple[datetime, float]:
    """
    Initializes the timer with Berlin timezone.
    Returns:
        start_dt: Wall clock time for logging (human readable).
        start_perf: Monotonic clock time for accurate duration measurement (unaffected by system clock updates).
    """
    tz = ZoneInfo("Europe/Berlin")
    start_dt = datetime.now(tz)
    start_perf = time.perf_counter()
    return start_dt, start_perf

def stop_timer(start_dt: datetime, start_perf: float) -> tuple[str, str]:
    """Calculates duration and returns formatted start time and duration strings."""
    # Use perf_counter for precision and monotonicity (immune to NTP/wall-clock changes)
    end_perf = time.perf_counter()
    duration_sec = end_perf - start_perf
    
    m, s = divmod(duration_sec, 60)
    h, m = divmod(m, 60)
    duration_str = f"{int(h):d}:{int(m):02d}:{int(s):02d}"
    
    start_time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    return start_time_str, duration_str
