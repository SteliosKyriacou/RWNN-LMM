"""Individual representation for evolutionary optimization."""

from __future__ import annotations
import copy
from typing import List, Optional

import numpy as np


class Individual:
    """A candidate solution in the evolutionary algorithm.

    Attributes:
        dofs: Design variables (normalized to [0,1]).
        scaleddofs: Scaled (physical) design variables.
        objs: Objective values after evaluation.
        pureobjs: Objective values before constraint penalties.
        constraints: Constraint function values.
        constraint_violation: Whether any constraint is violated.
        exactEvaluation: Whether this individual has been exactly evaluated.
        promptedobjs: Target objectives for GP inverse crossover.
        Auxiliary: Optional auxiliary data from evaluation.
    """

    __slots__ = (
        "dofs", "scaleddofs", "objs", "pureobjs", "constraints",
        "constraint_violation", "exactEvaluation", "promptedobjs", "Auxiliary",
    )

    def __init__(self) -> None:
        self.dofs: np.ndarray = np.array([])
        self.scaleddofs: np.ndarray = np.array([])
        self.objs: List[float] = []
        self.pureobjs: List[float] = []
        self.constraints: List[float] = []
        self.constraint_violation: bool = False
        self.exactEvaluation: bool = False
        self.promptedobjs: Optional[List[float]] = None
        self.Auxiliary = None

    def setdofs(self, dofs) -> None:
        self.dofs = np.asarray(dofs, dtype=np.float64)

    def setscaleddofs(self, scaleddofs) -> None:
        self.scaleddofs = np.asarray(scaleddofs, dtype=np.float64)

    def setobjs(self, objs) -> None:
        self.objs = list(objs)
        if self.objs and self.objs[0] == float("inf"):
            self.constraint_violation = True

    def setpureobjs(self, objs) -> None:
        self.pureobjs = list(objs)

    def setconstraints(self, constraints) -> None:
        self.constraints = list(constraints)

    def setstatus(self, status: bool) -> None:
        self.exactEvaluation = status

    def setpromptedobjs(self, objs) -> None:
        self.promptedobjs = list(objs)

    def setAuxiliary(self, aux) -> None:
        self.Auxiliary = aux

    def copy(self) -> "Individual":
        return copy.deepcopy(self)
