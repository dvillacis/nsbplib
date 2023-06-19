import numpy as np
import trustregion
from scipy.optimize import BFGS
from scipy.optimize import OptimizeResult
import pandas as pd

def _to_array(x,lbl):
    try:
        if isinstance(x,float):
            x = [x]
        return np.asarray_chkfinite(x)
    except ValueError:
        raise ValueError('%s contains Nan/Inf values' % lbl)
    
def update_tr_radius(Delta, actual_reduction, predicted_reduction,
                     step_norm):
    """Update the radius of a trust region based on the cost reduction.

    Returns
    -------
    Delta : float
        New radius.
    ratio : float
        Ratio between actual and predicted reductions.
    """
    if predicted_reduction > 0:
        ratio = actual_reduction / predicted_reduction
    elif predicted_reduction == actual_reduction == 0:
        ratio = 1
    else:
        ratio = 0

    if ratio < 0.25:
        Delta = 0.25 * step_norm
    elif ratio > 0.75:
        Delta *= 2.0

    return Delta, ratio

def check_termination(dF, F, dx_norm, x_norm, ratio, ftol, xtol):
    """Check termination condition"""
    ftol_satisfied = dF < ftol * F and ratio > 0.25
    xtol_satisfied = dx_norm < xtol * (xtol + x_norm)

    if ftol_satisfied and xtol_satisfied:
        return 4
    elif ftol_satisfied:
        return 2
    elif xtol_satisfied:
        return 3
    else:
        return None
    
def print_header():
    print("\n{0:^15}{1:^15}{2:^15}{3:^15}{4:^15}{5:^15}{6:^15}{7:^15}"
          .format("Iteration", "Total nfev", "Cost", "Cost reduction",
                  "Step norm", "Optimality", "TR-Radius", "B Cond"))
    
def print_iteration(iteration, nfev, cost, cost_reduction,
                              step_norm, optimality,radius,Bcond):
    if cost_reduction is None:
        cost_reduction = " " * 15
    else:
        cost_reduction = "{0:^15.2e}".format(cost_reduction)

    if step_norm is None:
        step_norm = " " * 15
    else:
        step_norm = "{0:^15.2e}".format(step_norm)

    print("{0:^15}{1:^15}{2:^15.4e}{3}{4}{5:^15.2e}{6:^15.2e}{7:^15.2e}"
          .format(iteration, nfev, cost, cost_reduction,
                  step_norm, optimality,radius,Bcond))
    
def solve(
    upper_level_problem,
    x0,
    verbose=False,
    initial_radius=1.0,
    threshold_radius=1e-4,
    max_radius=1000,
    max_nfev=2000,
    xtol=1e-9,
    ftol=1e-9,
    radius_tol=1e-4
    ):
    ### Evolution records
    nfevs = []
    costs = []
    costs_red = []
    steps = []
    optimalities = []
    radiuses = []
    Bconds = []
    
    ###
    x0 = _to_array(x0,'x0')
    x = x0
    radius = initial_radius
    termination_status = None
    iteration = 0
    nfev = 0
    njev = 0
    n_reg_jev = 0
    # B = BFGS(init_scale=1.0)
    B = BFGS()
    B.initialize(len(x),'hess')
    njev += 1
    nfev += 1
    f,g = upper_level_problem(x)
    g0 = np.copy(g)
    print_header()
    print_iteration(iteration,nfev,f,0,0,np.linalg.norm(g),radius,np.linalg.cond(B.get_matrix()))
    
    # updating record
    nfevs.append(nfev)
    costs.append(f)
    costs_red.append(0)
    steps.append(0)
    optimalities.append(np.linalg.norm(g))
    radiuses.append(radius)
    Bconds.append(np.linalg.cond(B.get_matrix()))
    ###
    
    while True:
        sl = -x - 1e-9
        su = 1e9 - x
        sl = np.where(sl < 0, sl, -sl)
        
        s = trustregion.solve(g,B.get_matrix(),radius,sl=sl,su=su)
        s_norm = np.linalg.norm(s)
        x_ = x + s
        pred = -np.dot(g,s)-0.5*np.dot(s,B.dot(s))
        nfev += 1
        if radius >= threshold_radius:
            njev += 1
            # print(f'Evaluating at {x_}')
            f_,g_ = upper_level_problem(x_)
            # print(f_,g_)
        else:
            n_reg_jev += 1
            f_,g_ = upper_level_problem(x_,smooth=True)
        ared = f-f_
        radius, ratio = update_tr_radius(radius,ared,pred,s_norm)
        if radius>max_radius:
            radius = max_radius
            
        # Checking termination
        termination_status = check_termination(ared,f,s_norm,np.linalg.norm(x),ratio,ftol,xtol)
        
        if radius < radius_tol:
            termination_status = 3
            
        if nfev > max_nfev:
            termination_status = 5

        if termination_status is not None:
            break
        
        if ared > 0:
            
            if np.dot(s, g_-g) > 0:
                B.update(x_-x,g_-g)
            else:
                # print(np.dot(s, g_-g))
                B.init_scale = 1e-12
                B.initialize(len(x),'hess')
                B.update(x_-x,g-g_)
                # print(B.get_matrix())
            x = x_
            g = g_
            f = f_
        else:
            s = 0
            ared = 0
            radius *= 0.5
        iteration += 1
        Bcond = np.linalg.cond(B.get_matrix())
        print_iteration(iteration,nfev,f,ared,s_norm,np.linalg.norm(g),radius,Bcond)
        # updating record
        nfevs.append(nfev)
        costs.append(f)
        costs_red.append(ared)
        steps.append(s_norm)
        optimalities.append(np.linalg.norm(g))
        radiuses.append(radius)
        Bconds.append(Bcond)
        
    if termination_status is None:
        termination_status = 0
        
    mydict = {
        'nfev':nfevs,
        'cost':costs,
        'cost reduction':costs_red,
        'step norm': steps,
        'optimality': optimalities,
        'radius': radiuses,
        'B Cond': Bconds
    }

    return pd.DataFrame.from_dict(mydict) , OptimizeResult(
        x=x, fun=f, jac=g, optimality=np.linalg.norm(g), nfev=nfev, njev=njev, n_reg_jev=n_reg_jev,nit=iteration, status=termination_status)