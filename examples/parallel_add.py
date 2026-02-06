"""Inspect-derived smoke test task.

Adapted from Inspect (`https://github.com/UKGovernmentBEIS/inspect_ai`), example `examples/tool_use.py` (parallel_add task).
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import tool


@tool
def add():
    async def execute(x: int, y: int):
        """
        Add two numbers.

        Args:
            x: First number to add.
            y: Second number to add.

        Returns:
            The sum of the two numbers.
        """
        return x + y

    return execute


@task
def parallel_add():
    return Task(
        dataset=[
            Sample(
                input=(
                    "Please add the numbers 1+1 and 2+2, and then print the "
                    "results of those computations side by side as just two "
                    "numbers (with no additional text). You should use the add "
                    "tool to do this, and you should make the two required calls "
                    "to add in parallel so the results are computed faster."
                ),
                target=["2 4"],
            )
        ],
        solver=[use_tools([add()]), generate()],
        scorer=includes(),
    )
