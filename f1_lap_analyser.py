"""
F1 Lap Time Analyser
====================
Loads one race's lap data with FastF1 and prints each driver's best,
worst and average lap time, ignoring lap 1 and any lap that touches
the pit lane.

Run the analysis:   python f1_lap_analyser.py
Run the tests:      python f1_lap_analyser.py --test
"""

# ---------------------------------------------------------------------------
# 1. SETTINGS - edit these to pick a different race
# ---------------------------------------------------------------------------
YEAR = 2024
RACE = "Monza"     # a name ("Monza", "Silverstone", "Bahrain") or round number
SESSION = "R"      # "R" = race. ("Q" = qualifying, "FP1" = practice 1, ...)
TOP_N = 5          # how many drivers to show in the top-N list

# ---------------------------------------------------------------------------
# 2. IMPORTS
# ---------------------------------------------------------------------------
import os
import sys
import unittest

import pandas as pd   # installed automatically alongside FastF1
import fastf1


# ---------------------------------------------------------------------------
# 3. LOADING - fetch the lap data from FastF1
# ---------------------------------------------------------------------------
def load_laps(year, race, session_type):
    """Download (or read from cache) one session and return its laps table."""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)   # 2nd run onwards is instant

    session = fastf1.get_session(year, race, session_type)
    # We only need lap times - skip the heavy telemetry/weather downloads.
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    print(f"Loaded: {session.event['EventName']} {year} ({session.name})\n")
    return session.laps


# ---------------------------------------------------------------------------
# 4. FILTERING - throw away the laps we don't want
# ---------------------------------------------------------------------------
def filter_laps(laps):
    """Keep only 'clean' laps: not lap 1, no pit stop, and a recorded time."""
    not_lap_one = laps["LapNumber"] > 1
    not_pit_in = laps["PitInTime"].isna()    # didn't drive INTO the pits
    not_pit_out = laps["PitOutTime"].isna()  # didn't drive OUT of the pits
    has_time = laps["LapTime"].notna()       # some laps have no time (e.g. crash)
    return laps[not_lap_one & not_pit_in & not_pit_out & has_time]


# ---------------------------------------------------------------------------
# 5. CALCULATING - best / worst / average per driver
# ---------------------------------------------------------------------------
def summarise(laps):
    """Return one row per driver with lap count, best, worst and average."""
    summary = laps.groupby("Driver")["LapTime"].agg(
        Laps="count", Best="min", Worst="max", Average="mean"
    )
    return summary.sort_values("Best")      # fastest driver first


# ---------------------------------------------------------------------------
# 6. DISPLAY - turn times into "1:32.845" and print tables
# ---------------------------------------------------------------------------
def format_time(td):
    """Convert a pandas Timedelta into a 'M:SS.mmm' string."""
    total = td.total_seconds()
    minutes = int(total // 60)
    seconds = total - minutes * 60
    return f"{minutes}:{seconds:06.3f}"


def print_table(summary):
    print(f"{'Driver':<8}{'Laps':>5}{'Best':>11}{'Worst':>11}{'Average':>11}")
    print("-" * 46)
    for driver, row in summary.iterrows():
        print(f"{driver:<8}{row['Laps']:>5}"
              f"{format_time(row['Best']):>11}"
              f"{format_time(row['Worst']):>11}"
              f"{format_time(row['Average']):>11}")


def print_top(summary, n):
    print(f"\nTop {n} by best lap")
    print("-" * 30)
    leader = summary["Best"].iloc[0]
    for position, (driver, row) in enumerate(summary.head(n).iterrows(), start=1):
        gap = (row["Best"] - leader).total_seconds()
        gap_text = "" if position == 1 else f"  +{gap:.3f}s"
        print(f"{position}. {driver:<5}{format_time(row['Best'])}{gap_text}")


def main():
    laps = load_laps(YEAR, RACE, SESSION)
    clean = filter_laps(laps)
    print(f"Using {len(clean)} of {len(laps)} laps "
          f"(lap 1, pit laps and untimed laps removed)\n")
    summary = summarise(clean)
    print_table(summary)
    print_top(summary, TOP_N)


# ---------------------------------------------------------------------------
# 7. TESTS - use small made-up data, so no internet needed
# ---------------------------------------------------------------------------
def _fake_laps(rows):
    """Build a tiny laps table shaped like FastF1's.
    rows = (driver, lap_number, seconds, pit_in, pit_out)"""
    return pd.DataFrame({
        "Driver": [r[0] for r in rows],
        "LapNumber": [r[1] for r in rows],
        "LapTime": [pd.Timedelta(seconds=r[2]) if r[2] is not None else pd.NaT
                    for r in rows],
        "PitInTime": [pd.Timedelta(seconds=1) if r[3] else pd.NaT for r in rows],
        "PitOutTime": [pd.Timedelta(seconds=1) if r[4] else pd.NaT for r in rows],
    })


class TestAnalyser(unittest.TestCase):

    def setUp(self):
        self.laps = _fake_laps([
            # driver, lap, seconds, pit_in, pit_out
            ("VER", 1, 99.0, False, False),   # lap 1      -> dropped
            ("VER", 2, 92.0, False, False),
            ("VER", 3, 94.0, False, False),
            ("VER", 4, 110.0, True, False),   # pit in     -> dropped
            ("VER", 5, 115.0, False, True),   # pit out    -> dropped
            ("VER", 6, 93.0, False, False),
            ("HAM", 2, 91.5, False, False),
            ("HAM", 3, 95.5, False, False),
            ("HAM", 4, None, False, False),   # no time    -> dropped
        ])

    def test_format_time(self):
        self.assertEqual(format_time(pd.Timedelta(seconds=92.845)), "1:32.845")
        self.assertEqual(format_time(pd.Timedelta(seconds=65.004)), "1:05.004")

    def test_filter_removes_lap_one_pits_and_untimed(self):
        clean = filter_laps(self.laps)
        self.assertNotIn(1, clean["LapNumber"].tolist())
        self.assertEqual(len(clean), 5)          # 3 VER + 2 HAM

    def test_best_worst_average(self):
        summary = summarise(filter_laps(self.laps))
        ver = summary.loc["VER"]
        self.assertEqual(ver["Laps"], 3)
        self.assertEqual(ver["Best"].total_seconds(), 92.0)
        self.assertEqual(ver["Worst"].total_seconds(), 94.0)
        self.assertEqual(ver["Average"].total_seconds(), 93.0)

    def test_sorted_fastest_first(self):
        summary = summarise(filter_laps(self.laps))
        self.assertEqual(list(summary.index), ["HAM", "VER"])  # 91.5 < 92.0


# ---------------------------------------------------------------------------
# 8. ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if "--test" in sys.argv:
        unittest.main(argv=[sys.argv[0], "-v"])
    else:
        main()
