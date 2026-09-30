"""
Anytime-bound recording for the GIP solvers, without log-file parsing.

Solver-side use (two lines per RunSolver):

    from Analysis.gurobi_trace import AnytimeTrace
    trace = AnytimeTrace()
    m.optimize(trace.wrap(cut_heuristic_callback))     # or trace.wrap(None)
    if stats_out is not None:
        stats_out.update(trace.summary(m))

`trace.wrap` chains your own callback unchanged; it only reads MIP_OBJBST /
MIP_OBJBND / RUNTIME and appends a point whenever either bound changes.
"""
from __future__ import annotations

import math


def _clean(x: float) -> float:
    """Gurobi reports +/-1e100 (GRB.INFINITY) before a bound exists."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) and abs(x) < 1e99 else float("nan")


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


class AnytimeTrace:
    """Collects (time, incumbent, best bound) along a Gurobi MIP solve."""

    def __init__(self):
        self.t: list[float] = []
        self.ub: list[float] = []
        self.lb: list[float] = []

    def _push(self, t: float, ub: float, lb: float):
        ub, lb = _clean(ub), _clean(lb)
        if self.t and _same(self.ub[-1], ub) and _same(self.lb[-1], lb):
            return
        self.t.append(float(t))
        self.ub.append(ub)
        self.lb.append(lb)

    def wrap(self, user_callback=None):
        from gurobipy import GRB

        def _cb(model, where):
            if where == GRB.Callback.MIP:
                self._push(model.cbGet(GRB.Callback.RUNTIME),
                           model.cbGet(GRB.Callback.MIP_OBJBST),
                           model.cbGet(GRB.Callback.MIP_OBJBND))
            elif where == GRB.Callback.MIPNODE:
                self._push(model.cbGet(GRB.Callback.RUNTIME),
                           model.cbGet(GRB.Callback.MIPNODE_OBJBST),
                           model.cbGet(GRB.Callback.MIPNODE_OBJBND))
            if user_callback is not None:
                user_callback(model, where)

        return _cb

    def summary(self, model) -> dict:
        """Final solver statistics + the trace, ready for records.save_run(solve=...)."""
        from gurobipy import GRB

        status_names = {getattr(GRB, n): n for n in dir(GRB)
                        if n.isupper() and isinstance(getattr(GRB, n), int)
                        and n in ("LOADED", "OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED",
                                  "CUTOFF", "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT",
                                  "SOLUTION_LIMIT", "INTERRUPTED", "NUMERIC", "SUBOPTIMAL",
                                  "INPROGRESS", "USER_OBJ_LIMIT", "WORK_LIMIT", "MEM_LIMIT")}
        sol_count = int(model.SolCount)
        objective = _clean(model.ObjVal) if sol_count > 0 else float("nan")
        try:
            bound = _clean(model.ObjBound)
        except Exception:
            bound = float("nan")
        try:
            gap = _clean(model.MIPGap) if sol_count > 0 else float("nan")
        except Exception:
            gap = float("nan")
        runtime = float(model.Runtime)

        # close the trace with the final state
        self._push(runtime, objective, bound)

        first = next((t for t, u in zip(self.t, self.ub) if not math.isnan(u)), float("nan"))
        return {
            "status": status_names.get(model.Status, str(model.Status)),
            "runtime_s": runtime,
            "objective": objective,
            "bound": bound,
            "gap": gap,
            "node_count": float(getattr(model, "NodeCount", float("nan"))),
            "sol_count": sol_count,
            "num_vars": int(model.NumVars),
            "num_constrs": int(model.NumConstrs),
            "time_limit": float(model.Params.TimeLimit),
            "time_to_first_s": first,
            "trace": {"t": list(self.t), "ub": list(self.ub), "lb": list(self.lb)},
        }
