"""Inspect-derived smoke test task.

Adapted from Inspect (`https://github.com/UKGovernmentBEIS/inspect_ai`), example `examples/hello_world.py`.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate


@task
def hello_world():
    return Task(
        dataset=[
            Sample(
                input="Just reply with Hello World",
                target="Hello World",
            )
        ],
        solver=[generate()],
        scorer=exact(),
    )
