import torch

from torch_utils.torch_utils import get_device

def cg_solver(Avp_fun, b, max_iter=10):
    '''
    Finds an approximate solution to a set of linear equations Ax = b

    Parameters
    ----------
    Avp_fun : callable
        a function that right multiplies a matrix A by a vector

    b : torch.FloatTensor
        the right hand term in the set of linear equations Ax = b

    max_iter : int
        the maximum number of iterations (default is 10)

    Returns
    -------
    x : torch.FloatTensor
        the approximate solution to the system of equations defined by Avp_fun
        and b
    '''

    device = get_device()
    x = torch.zeros_like(b).to(device)
    r = b.clone()
    p = b.clone()

    for i in range(max_iter):
        Avp = Avp_fun(p, retain_graph=True)

        alpha = torch.matmul(r, r) / torch.matmul(p, Avp)
        x += alpha * p

        if i == max_iter - 1:
            return x

        r_new = r - alpha * Avp
        beta = torch.matmul(r_new, r_new) / torch.matmul(r, r)
        r = r_new
        p = r + beta * p
