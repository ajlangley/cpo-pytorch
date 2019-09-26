import torch
from torch.autograd import grad

from torch_utils.torch_utils import flat_grad

def get_Hvp_fun(functional_output, inputs, damping_coef=0.0):
    '''
    Returns a function that calculates a Hessian-vector product with the Hessian
    of functional_output w.r.t. inputs

    Parameters
    ----------
    functional_output : torch.FloatTensor (with requires_grad=True)
        the output of the function of which the Hessian is calculated

    inputs : torch.FloatTensor
        the inputs w.r.t. which the Hessian is calculated

    damping_coef : float
        the multiple of the identity matrix to be added to the Hessian
    '''

    inputs = list(inputs)
    grad_f = flat_grad(functional_output, inputs, create_graph=True)

    def Hvp_fun(v, retain_graph=True):
        gvp = torch.matmul(grad_f, v)
        Hvp = flat_grad(gvp, inputs, retain_graph=retain_graph)
        Hvp += damping_coef * v

        return Hvp

    return Hvp_fun
