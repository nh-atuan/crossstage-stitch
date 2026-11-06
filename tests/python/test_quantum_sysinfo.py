"""Q11 resource log: peak RSS and RAM must be real readings, not a silent 0."""
from tools.quantum.sysinfo import peak_rss_bytes, total_ram_bytes


def test_memory_readings_are_positive_and_ordered():
    peak, total = peak_rss_bytes(), total_ram_bytes()
    assert 0 < peak < total
