"""Inspect-derived smoke test tasks.

Adapted from Inspect (`https://github.com/UKGovernmentBEIS/inspect_ai`), example `examples/hello_world.py`.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate


@task
def hello_world_a() -> Task:
    return Task(
        dataset=[
            Sample(
                input="Reply with exactly: Hello World",
                target="Hello World",
            )
        ],
        solver=[generate()],
        scorer=exact(),
    )


@task
def hello_world_b() -> Task:
    return Task(
        dataset=[
            Sample(
                input="Reply with just two words in title case: hello world",
                target="Hello World",
            )
        ],
        solver=[generate()],
        scorer=exact(),
    )
