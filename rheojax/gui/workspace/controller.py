from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Step:
    id: str
    title: str
    is_ready: Callable[[], bool]
    validate: Callable[[], bool]


class WorkflowController:
    def __init__(self, steps: list[Step]) -> None:
        self.steps = steps
        self.current = 0
        self.reached: set[int] = {0}
        self.revision = 0

    def can_advance(self) -> bool:
        return (
            self.current + 1 < len(self.steps)
            and self.steps[self.current].is_ready()
            and self.steps[self.current].validate()
        )

    def advance(self) -> None:
        if self.can_advance():
            self.current += 1
            self.reached.add(self.current)

    def goto(self, index: int) -> bool:
        if index in self.reached:
            self.current = index
            return True
        return False

    def on_edit(self, step_index: int) -> None:
        """Editing a completed step bumps revision and re-locks everything downstream."""
        if not 0 <= step_index < len(self.steps):
            raise ValueError(
                f"step_index {step_index} out of range [0, {len(self.steps)})"
            )
        self.revision += 1
        self.reached = {i for i in self.reached if i <= step_index}
        if self.current > step_index:
            self.current = step_index


class FitController(WorkflowController):
    STEP_IDS = ["protocol_model", "data", "nlsq", "nuts", "visualize", "export"]


class TransformController(WorkflowController):
    STEP_IDS = ["transform", "slots", "run", "visualize", "export"]
