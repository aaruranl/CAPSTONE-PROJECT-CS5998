"""Tests for the log feed (src/replay.py): finding the start time and pacing the lines."""
from src.replay import batches, seek_time


def first_field(raw: bytes):
    head = raw.split(maxsplit=1)[0] if raw.strip() else b""
    return int(head) if head.isdigit() else None


def test_seek_time_finds_the_first_line_at_or_after_a_time(tmp_path):
    path = tmp_path / "log.txt"
    path.write_bytes(b"10 a\n20 b\n20 c\n?? unreadable\n30 d\n40 e\n")
    with open(path, "rb") as f:
        for t, expected in [(0, b"10 a\n"), (10, b"10 a\n"), (11, b"20 b\n"), (20, b"20 b\n"),
                            (25, b"?? unreadable\n"), (31, b"40 e\n"), (99, b"")]:
            seek_time(f, t, first_field)
            assert f.readline() == expected, t


class FakeClock:
    """Pretends to be time.monotonic / time.sleep, so the test does not really wait."""
    def __init__(self):
        self.now, self.sleeps = 100.0, []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(round(seconds, 6))
        self.now += seconds


def test_batches_backfill_fast_then_follow_log_time():
    records = [(5, "a"), (8, "b"),                    # back-fill (before go-live 10): no waiting
               (10, "c"), (10, "d"), (11, "e"),       # due within one TICK: one request
               (13, "f"),                            # due 3 s later: wait, with a heartbeat every second
               (500, "g")]                            # a long silence: shortened to max_wait
    fake = FakeClock()
    out = list(batches(records, go_live=10, speed=1, batch_size=100, max_wait=5,
                       clock=fake.clock, sleep=fake.sleep))
    assert out == [(["a", "b"], None),
                   (["c", "d", "e"], 10), ([], 11), ([], 12),           # heartbeats tell the API the time
                   (["f"], 13), ([], 496), ([], 497), ([], 498), ([], 499),  # skipped ahead; never past 499
                   (["g"], None)]
    assert fake.sleeps == [1.0] * 8


def test_batches_speed_zero_only_limits_the_batch_size():
    fake = FakeClock()
    records = [(t, str(t)) for t in range(5)]
    out = list(batches(records, go_live=0, speed=0, batch_size=2, clock=fake.clock, sleep=fake.sleep))
    assert out == [(["0", "1"], None), (["2", "3"], None), (["4"], None)] and fake.sleeps == []


def test_batches_faster_speed_means_shorter_pauses():
    fake = FakeClock()
    out = list(batches([(0, "a"), (600, "b")], go_live=0, speed=300, batch_size=100, max_wait=10,
                       clock=fake.clock, sleep=fake.sleep))
    assert fake.sleeps == [1.0, 1.0]   # 600 log seconds at 300x = 2 real seconds
    assert out == [(["a"], 0), ([], 300), (["b"], None)]  # after 1 real second it is 300 s later in the log
