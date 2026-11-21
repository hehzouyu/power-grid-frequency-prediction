"""Score the R5 spatial holdout with the frozen R3 thresholds."""
from pathlib import Path
import argparse

from score_round4 import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition", action="store_true")
    out = Path(__file__).resolve().parent / "out" / "round5"
    if parser.parse_args().transition:
        run(out / "transition", range(2101, 2133), 39195021, report_patterns=True)
    else:
        run(out, range(2001, 2049), 39195020, report_patterns=True)
