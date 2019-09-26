import torch

def line_search(search_dir, max_step_len, constraints_satisfied, line_search_coef=0.9,
                max_iter=10):
    '''
    Perform a backtracking line search that terminates when constraints_satisfied
    return True and return the calculated step length. Return 0.0 if no step
    length can be found for which constraints_satisfied returns True

    Parameters
    ----------
    search_dir : torch.FloatTensor
        the search direction along which the line search is done

    max_step_len : torch.FloatTensor
        the maximum step length to consider in the line search

    constraints_satisfied : callable
        a function that returns a boolean indicating whether the constraints
        are met by the current step length

    line_search_coef : float
        the proportion by which to reduce the step length after each iteration

    max_iter : int
        the maximum number of backtracks to do before return 0.0

    Returns
    -------
    the maximum step length coefficient for which constraints_satisfied evaluates
    to True
    '''

    step_len = max_step_len / line_search_coef

    for i in range(max_iter):
        step_len *= line_search_coef

        if constraints_satisfied(step_len * search_dir, step_len):
            return step_len

    return torch.tensor(0.0)
