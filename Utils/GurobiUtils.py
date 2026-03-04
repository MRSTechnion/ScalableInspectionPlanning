import gurobipy as gp
from gurobipy import GRB, quicksum
import CutsOracle

def explain_infeasibility_of_point(m, x_by_varname, int_tol=1e-5):
    """
    m: original model (already built)
    x_by_varname: dict {var.VarName: value} describing the candidate solution
    """
    # Copy the model so we don't disturb the original
    mc = m.copy()
    mc.Params.OutputFlag = 0

    # Fix all variables to the proposed values (respecting integrality)
    for v in mc.getVars():
        xv = x_by_varname.get(v.VarName, None)
        if xv is None:
            # If you didn't supply a value for every var, that alone can cause rejection.
            raise ValueError(f"Missing value for variable {v.VarName}")
        if v.VType in (GRB.BINARY, GRB.INTEGER):
            xv = round(xv) if abs(xv - round(xv)) <= int_tol else xv
        v.LB = xv
        v.UB = xv

    # Try to optimize the feasibility problem (objective doesn't matter)
    mc.setObjective(0.0)
    mc.optimize()

    if mc.Status == GRB.OPTIMAL:
        print("The point is feasible (all constraints satisfied at Gurobi tolerances).")
        return

    if mc.Status not in (GRB.INFEASIBLE,):
        print(f"Unexpected status {mc.Status}. (INF_OR_UNBD? Try setting DualReductions=0 and re-run.)")
        mc.Params.DualReductions = 0
        mc.optimize()

    if mc.Status == GRB.INFEASIBLE:
        # Compute IIS to see the smallest irreducible infeasible subsystem
        mc.computeIIS()

        print("\nIIS report (constraints/bounds that cannot all be satisfied together):")
        # Marked constraints
        for c in mc.getConstrs():
            if c.IISConstr:
                print(f"  Constr: {c.ConstrName}")
        # Marked quadratic constraints
        for qc in mc.getQConstrs():
            if qc.IISQConstr:
                print(f"  QConstr: {qc.QCName}")
        # Marked general constraints (indicators, etc.)
        for gc in mc.getGenConstrs():
            if gc.IISGenConstr:
                print(f"  GenConstr: {gc.GenConstrName}")
        # Variable bounds in the IIS
        for v in mc.getVars():
            if v.IISLB:
                print(f"  Var LB: {v.VarName} = {v.LB}")
            if v.IISUB:
                print(f"  Var UB: {v.VarName} = {v.UB}")



def check_feasibility_in_model(model, sol, tol=1e-6, max_violations=20, name="model1"):
    """
    Check feasibility of a candidate solution (given as var_name->value)
    in 'model', by evaluating each constraint.

    Returns (is_feasible, max_violation).
    """
    print(f"\n=== Checking feasibility of model2 solution in {name} ===")

    # Build a quick lookup for speed
    val = sol

    max_viol = 0.0
    viol_list = []

    for c in model.getConstrs():
        row = model.getRow(c)
        activity = 0.0
        for i in range(row.size()):
            var = row.getVar(i)
            coeff = row.getCoeff(i)
            vname = var.VarName
            if vname in val:
                activity += coeff * val[vname]
            else:
                # If variable is missing in solution, treat as 0
                # (or you can choose to raise an error)
                pass

        rhs = c.RHS
        sense = c.Sense

        if sense == '<':
            viol = activity - rhs
        elif sense == '>':
            viol = rhs - activity
        elif sense == '=':
            viol = abs(activity - rhs)
        else:
            raise RuntimeError(f"Unknown constraint sense {sense}")

        if viol > max_viol:
            max_viol = viol
        if viol > tol:
            viol_list.append((c.ConstrName, viol, activity, rhs, sense))

    if max_viol <= tol:
        print(f"{name}: candidate solution is feasible within tolerance {tol}.")
    else:
        print(f"{name}: candidate solution is INFEASIBLE. Max violation = {max_viol:.3e}")
        print(f"Showing up to {max_violations} largest violations (by raw violation):")
        # sort by violation descending
        viol_list.sort(key=lambda x: -x[1])
        for cname, viol, act, rhs, sense in viol_list[:max_violations]:
            print(f"  {cname}: viol={viol:.3e}, sense={sense}, activity={act}, rhs={rhs}")

    return max_viol <= tol, max_viol



def solve_with_callback(model, callback=None, name="model"):
    """Solve a model with an optional callback. Return solution dict (var_name -> value)."""
    print(f"=== Solving {name} ===")
    model.optimize(callback)

    status = model.Status
    if status == GRB.OPTIMAL:
        print(f"{name}: optimal, obj = {model.ObjVal}")
    else:
        print(f"{name}: status = {status}")
        if model.SolCount > 0:
            print(f"  incumbent = {model.ObjVal}, bound = {model.ObjBound}, MIPGap = {model.MIPGap}")
        else:
            print("  no feasible solution")

    if model.SolCount == 0:
        return {}

    return {v.VarName: v.X for v in model.getVars()}


def inject_as_mip_start(model, sol):
    """
    Inject solution into 'model' as a MIP start and allow Gurobi
    to try to accept it as an incumbent.
    """

    missing = 0
    for v in model.getVars():
        if v.VarName in sol:
            v.Start = sol[v.VarName]
        else:
            missing += 1

    if missing > 0:
        print(f"Error: couldn't inject ")

    model.Params.StartNodeLimit = 0
    model.optimize()

    if model.SolCount == 0:
        print(f"{name}: MIP start was rejected.")
        return None
    else:
        print(f"{name}: MIP start accepted. Incumbent obj = {model.ObjVal}")
        return model.ObjVal


def compare_bc_models(model1, model2, callback1=None, callback2=None):
    """
    Compare two branch-and-cut models (with callbacks).

    - Solve model2 with callback2.
    - Take its incumbent solution.
    - Inject as MIP start into model1, solve with callback1.
    - See whether it is feasible & what obj it gives.

    callback1/callback2 can be the same function if both use the same logic.
    """

    # Sanity: same variable names?
    vars1 = {v.VarName for v in model1.getVars()}
    vars2 = {v.VarName for v in model2.getVars()}
    if vars1 != vars2:
        print("WARNING: variable sets differ between model1 and model2!")
        print(f"  only in model1: {len(vars1 - vars2)}")
        print(f"  only in model2: {len(vars2 - vars1)}")

    # 1) Solve model2 with its BnC logic
    sol2 = solve_with_callback(model2, callback2, name="model2")
    if not sol2:
        print("model2 produced no solution, aborting comparison.")
        return {}

    obj2 = model2.ObjVal

    # 2) Inject into model1 and see if it survives its own BnC logic
    obj1_from_start = inject_as_mip_start_with_callback(
        model1, sol2, callback1, name="model1"
    )

    return {
        "model2_obj": obj2,
        "model1_obj_from_model2_start": obj1_from_start,
    }


if __name__ == '__main__':
    model1_path = "/home/adir/Desktop/IP-results/models/drone_1000.lp"
    model2_path = "/home/adir/Desktop/IP-results/models/rootInflow_drone_1000.lp"

    m1 = gp.read(model1_path)
    m2 = gp.read(model2_path)

    results = compare_bc_models(m1, m2,
                                callback1=cut_heuristic_callback, callback2=cut_heuristic_callback)

    print(results)
