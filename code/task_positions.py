"""Split a position spec into blocks for SLURM array tasks.

    python code/task_positions.py <spec> <task> <per_task>   # task's ids, comma-separated
    python code/task_positions.py <spec> --array <per_task>  # the --array range, e.g. 0-14

Task i takes the i-th block of ``per_task`` positions from the spec's list
(registry order, see positions.py). A task beyond the end prints nothing.
"""

from __future__ import annotations

import math
import sys

from positions import parse_position_spec


def block(spec: str, task: int, per_task: int) -> list[str]:
    ids = parse_position_spec(spec)
    return ids[task * per_task:(task + 1) * per_task]


def array_range(spec: str, per_task: int) -> str:
    n = math.ceil(len(parse_position_spec(spec)) / per_task)
    return f"0-{n - 1}"


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "--array":
        print(array_range(argv[0], int(argv[2])))
        return 0
    if len(argv) != 3:
        print(__doc__)
        return 2
    print(",".join(block(argv[0], int(argv[1]), int(argv[2]))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
