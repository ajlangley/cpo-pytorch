import torch


class RunningStat:
    '''
    Keeps track of a running estimate of the mean and standard deviation of
    a distribution based on the observations seen so far

    Attributes
    ----------
    _M : torch.float
        estimate of the mean of the observations seen so far

    _S : torch.float
        estimate of the sum of the squared deviations from the mean of the
        observations seen so far

    n : int
        the number of observations seen so far

    Methods
    -------
    update(x)
        update the running estimates of the mean and standard deviation

    mean()
        return the estimated mean

    var()
        return the estimated variance

    std()
        return the estimated standard deviation
    '''

    def __init__(self):
        self._M = None
        self._S = None
        self.n = 0

    def update(self, x):
        self.n += 1

        if self.n == 1:
            self._M = x.clone()
            self._S = torch.zeros_like(x)
        else:
            old_M = self._M.clone()
            self._M = old_M + (x - old_M) / self.n
            self._S = self._S + (x - old_M) * (x - self._M)

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        if self.n > 1:
            var = self._S / (self.n - 1)
        else:
            var = torch.pow(self.mean, 2)

        return var

    @property
    def std(self):
        return torch.sqrt(self.var)
