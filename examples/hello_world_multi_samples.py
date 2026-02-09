"""Inspect-derived smoke test task with multiple samples.

Adapted from Inspect (`https://github.com/UKGovernmentBEIS/inspect_ai`), example `examples/hello_world.py`.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate


@task
def hello_world_multi_samples() -> Task:
    return Task(
        dataset=[
            Sample(
                input="Reply with exactly: Hello World",
                target="Hello World",
            ),
            Sample(
                input="Write only these two words in title case: hello world",
                target="Hello World",
            ),
            Sample(
                input="Output just the phrase Hello World and nothing else",
                target="Hello World",
            ),
        ],
        solver=[generate()],
        scorer=exact(),
    )
