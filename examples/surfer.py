"""Inspect-derived smoke test task.

Adapted from Inspect (`https://github.com/UKGovernmentBEIS/inspect_ai`), example `examples/surfer.py`.
"""

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.tool import web_search


@task
def surfer() -> Task:
    return Task(
        dataset=[Sample(input="What were the scores of last night's NHL games?")],
        solver=react(tools=[web_search()]),
    )
